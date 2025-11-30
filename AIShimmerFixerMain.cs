using System.Diagnostics;

using OpenCvSharp;

namespace DigitalRuby.AIShimmerFixer;

/// <summary>
/// Fix shimmering in AI-generated videos using advanced Temporal Anti-Aliasing (TAA) techniques.
/// Args: input video path, output video path
/// </summary>
public static class AIShimmerFixerMain
{
    // --- TAA CONFIGURATION (B′ + Flow Confidence Tweaks) ---

    // 1. BLEND LIMITS
    // Slightly lower MinBlend than B (0.10 -> 0.08) for more history on static pixels,
    // but still high enough to avoid mush.
    private const double MinBlend = 0.08;
    private const double MaxBlend = 1.0;

    // 2. OCCLUSION SENSITIVITY
    // Slightly looser than B (0.5 -> 0.55) to reduce over-eager history resets,
    // which can cause shimmer at edges.
    private const double ConsistencyThreshold = 0.55;

    // 3. VELOCITY WEIGHTING (Balanced)
    private const double MinVelocityThreshold = 0.5;
    private const double MaxVelocityThreshold = 10.0;

    // 4. MOTION DILATION
    private const int MotionDilationSize = 3;

    // 5. SHARPENING
    private const double SharpenAmount = 0.0;

    // 6. FLOW SETTINGS
    private const double FlowScale = 0.5; // 0.5 for speed (dual-pass is heavy)
    private const int FlowWindowSize = 25;
    private const int FlowPolyN = 7;
    private const double FlowPolySigma = 1.5;

    // 7. MOTION MASK (Balanced)
    // Pixels with flow magnitude above this are considered "in motion".
    // In those areas, we enforce a higher minimum alpha so we trust the current frame more.
    private const double MotionThreshold = 0.3;   // px/frame
    private const double MotionMinAlpha = 0.6;    // enforce alpha >= 0.6 in motion regions

    // 8. STATIC-REGION BIAS
    // Very low motion → slightly *lower* alpha (more history) to reduce shimmer.
    private const double StaticVelocityThreshold = 0.15; // px/frame
    private const double StaticAlphaScale = 0.8;         // scale alpha by 0.8 in static regions

    // 9. FLOW CONFIDENCE LIMITS
    // Below FlowStaticEpsilon: treat flow as zero (no warp) to avoid tiny warping jitter.
    // Above MaxTrustedFlow: treat flow as untrusted and fall back to current frame.
    private const double FlowStaticEpsilon = 0.05; // px/frame
    private const double MaxTrustedFlow = 20.0;    // px/frame

    static void Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Usage: TaaProcessor.exe <input.mp4> <output.mp4>");
            return;
        }

        string inputPath = args[0];
        string outputPath = args[1];
        string tempVideoPath = "temp_video_no_audio.mp4";

        if (!File.Exists(inputPath))
        {
            Console.WriteLine($"Error: Input file '{inputPath}' not found.");
            return;
        }

        Console.WriteLine("--- Starting TAA Processing (B′ + Flow Confidence) ---");
        Console.WriteLine("Technique: Dual-Pass Occlusion Detection + AABB Clamping + Motion-Aware Alpha");
        Console.WriteLine($"[Config] Consistency Threshold: {ConsistencyThreshold}px");
        Console.WriteLine($"[Config] Velocity Range: {MinVelocityThreshold} .. {MaxVelocityThreshold} px/frame");
        Console.WriteLine($"[Config] Motion Threshold: {MotionThreshold}px, MotionMinAlpha: {MotionMinAlpha}");
        Console.WriteLine($"[Config] StaticVelocityThreshold: {StaticVelocityThreshold}px, StaticAlphaScale: {StaticAlphaScale}");
        Console.WriteLine($"[Config] FlowStaticEpsilon: {FlowStaticEpsilon}px, MaxTrustedFlow: {MaxTrustedFlow}px");
        Console.WriteLine($"[Config] MinBlend: {MinBlend}, MaxBlend: {MaxBlend}");

        ProcessVideo(inputPath, tempVideoPath);

        Console.WriteLine("\n--- Merging Audio ---");
        bool audioSuccess = MuxAudio(inputPath, tempVideoPath, outputPath);

        if (File.Exists(tempVideoPath)) File.Delete(tempVideoPath);

        if (audioSuccess)
            Console.WriteLine($"\nSuccess! Output saved to: {outputPath}");
        else
            Console.WriteLine("\nFinished with errors. Check FFmpeg output.");
    }

    static void ProcessVideo(string inputFile, string tempOutputFile)
    {
        using (var capture = new VideoCapture(inputFile))
        {
            if (!capture.IsOpened())
                throw new Exception("Could not open video file.");

            int width = (int)capture.Get(VideoCaptureProperties.FrameWidth);
            int height = (int)capture.Get(VideoCaptureProperties.FrameHeight);
            double fps = capture.Get(VideoCaptureProperties.Fps);
            int totalFrames = (int)capture.Get(VideoCaptureProperties.FrameCount);

            using (var writer = new VideoWriter(tempOutputFile, FourCC.MP4V, fps, new Size(width, height)))
            {
                // --- GPU CONTAINERS (UMat) ---
                UMat currentFrame = new UMat();
                UMat currentGray = new UMat();
                UMat prevGray = new UMat();
                UMat accumulationBuffer = new UMat();
                UMat finalOutput = new UMat();

                // Logic Buffers
                UMat currentFloat = new UMat();
                UMat historyWarped = new UMat();
                UMat flowFwd = new UMat(); // Prev -> Curr

                UMat minNeighbor = new UMat();
                UMat maxNeighbor = new UMat();

                // Maps
                UMat mapX = new UMat(new Size(width, height), MatType.CV_32F);
                UMat mapY = new UMat(new Size(width, height), MatType.CV_32F);

                // Kernels
                Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
                Mat dilationKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(MotionDilationSize, MotionDilationSize));

                // Sharpening Kernel
                Mat sharpenKernel = new Mat(3, 3, MatType.CV_32F);
                float s = (float)SharpenAmount;
                float center = 1.0f + (4.0f * s);
                float neighbor = -s;
                sharpenKernel.Set(0, 0, 0f); sharpenKernel.Set(0, 1, neighbor); sharpenKernel.Set(0, 2, 0f);
                sharpenKernel.Set(1, 0, neighbor); sharpenKernel.Set(1, 1, center); sharpenKernel.Set(1, 2, neighbor);
                sharpenKernel.Set(2, 0, 0f); sharpenKernel.Set(2, 1, neighbor); sharpenKernel.Set(2, 2, 0f);

                CreateRemapGrid(width, height, mapX, mapY);

                int frameIndex = 0;

                // CPU Temps for Flow
                Mat cpuPrev = new Mat();
                Mat cpuCurr = new Mat();
                Mat cpuPrevSmall = new Mat();
                Mat cpuCurrSmall = new Mat();
                Mat cpuFlowFwdSmall = new Mat();
                Mat cpuFlowBwdSmall = new Mat();
                Mat cpuFlowFwd = new Mat();
                Mat cpuFlowBwd = new Mat();

                // Occlusion Check Mats
                Mat consistencyError = new Mat();
                Mat consistencyMask = new Mat();
                UMat gpuConsistencyMask = new UMat();

                // High-velocity mask (for untrusted large motion)
                UMat gpuBigMask = new UMat();

                while (capture.Read(currentFrame))
                {
                    if (currentFrame.Empty())
                        break;

                    Cv2.CvtColor(currentFrame, currentGray, ColorConversionCodes.BGR2GRAY);
                    currentFrame.ConvertTo(currentFloat, MatType.CV_32FC3);

                    if (frameIndex == 0)
                    {
                        // First frame: initialize accumulation buffer directly.
                        currentFloat.CopyTo(accumulationBuffer);
                    }
                    else
                    {
                        // 1. DUAL OPTICAL FLOW (Forward & Backward)
                        prevGray.CopyTo(cpuPrev);
                        currentGray.CopyTo(cpuCurr);

                        // Resize for speed (FlowScale)
                        Cv2.Resize(cpuPrev, cpuPrevSmall, new Size(0, 0), FlowScale, FlowScale);
                        Cv2.Resize(cpuCurr, cpuCurrSmall, new Size(0, 0), FlowScale, FlowScale);

                        // Flow FWD (Prev -> Curr)
                        Cv2.CalcOpticalFlowFarneback(
                            cpuPrevSmall, cpuCurrSmall, cpuFlowFwdSmall,
                            0.5, 3, FlowWindowSize, 3, FlowPolyN, FlowPolySigma,
                            OpticalFlowFlags.None
                        );

                        // Flow BWD (Curr -> Prev)
                        Cv2.CalcOpticalFlowFarneback(
                            cpuCurrSmall, cpuPrevSmall, cpuFlowBwdSmall,
                            0.5, 3, FlowWindowSize, 3, FlowPolyN, FlowPolySigma,
                            OpticalFlowFlags.None
                        );

                        // Resize back and scale to full resolution
                        Cv2.Resize(cpuFlowFwdSmall, cpuFlowFwd, new Size(width, height));
                        Cv2.Resize(cpuFlowBwdSmall, cpuFlowBwd, new Size(width, height));
                        cpuFlowFwd *= (1.0 / FlowScale);
                        cpuFlowBwd *= (1.0 / FlowScale);

                        // --- FLOW CONFIDENCE TWEAKS ---

                        // Compute forward flow magnitude
                        Mat flowMag = new Mat();
                        Mat fx = new Mat();
                        Mat fy = new Mat();
                        Cv2.ExtractChannel(cpuFlowFwd, fx, 0);
                        Cv2.ExtractChannel(cpuFlowFwd, fy, 1);
                        Cv2.Magnitude(fx, fy, flowMag);

                        // 2a) Zero *tiny* flows (avoid warping in almost-static areas)
                        // smallMask = 1 where mag > FlowStaticEpsilon (keep), 0 where mag <= epsilon (zero out)
                        Mat smallMask = new Mat();
                        Cv2.Threshold(flowMag, smallMask, FlowStaticEpsilon, 1.0, ThresholdTypes.Binary);

                        Cv2.Multiply(fx, smallMask, fx);
                        Cv2.Multiply(fy, smallMask, fy);

                        // Merge back into cpuFlowFwd
                        Cv2.Merge(new[] { fx, fy }, cpuFlowFwd);

                        // 2b) Build mask for *very large* flows (untrusted, use current frame later)
                        // bigMask = 1 where mag > MaxTrustedFlow
                        Mat bigMask = new Mat();
                        Cv2.Threshold(flowMag, bigMask, MaxTrustedFlow, 1.0, ThresholdTypes.Binary);
                        bigMask.CopyTo(gpuBigMask);

                        // 2. CONSISTENCY CHECK (Occlusion Detection)
                        // Ideally: FlowFWD(p) + FlowBWD(p + FlowFWD(p)) ~= 0.
                        // Simplified local check: |FlowFWD + FlowBWD| < Threshold ⇒ consistent.
                        Mat flowSum = new Mat();
                        Cv2.Add(cpuFlowFwd, cpuFlowBwd, flowSum); // Vector sum

                        // Calculate magnitude of the sum (Error)
                        Mat flowSumCh1 = new Mat();
                        Mat flowSumCh2 = new Mat();
                        Cv2.ExtractChannel(flowSum, flowSumCh1, 0);
                        Cv2.ExtractChannel(flowSum, flowSumCh2, 1);
                        Cv2.Magnitude(flowSumCh1, flowSumCh2, consistencyError);

                        // Generate Mask: 1.0 = Occluded/Bad, 0.0 = Consistent/Good
                        Cv2.Threshold(consistencyError, consistencyMask, ConsistencyThreshold, 1.0, ThresholdTypes.Binary);

                        // Dilate the Occlusion Mask slightly to catch edges
                        Cv2.Dilate(consistencyMask, consistencyMask, dilationKernel);

                        // Upload to GPU
                        cpuFlowFwd.CopyTo(flowFwd);
                        consistencyMask.CopyTo(gpuConsistencyMask);

                        // 3. REPROJECTION (Warp history into current frame space)
                        UpdateRemapMaps(mapX, mapY, flowFwd);
                        Cv2.Remap(accumulationBuffer, historyWarped, mapX, mapY, InterpolationFlags.Linear, BorderTypes.Reflect);

                        // 4. AABB CLAMPING (Standard Stability)
                        Cv2.Erode(currentFloat, minNeighbor, kernel);
                        Cv2.Dilate(currentFloat, maxNeighbor, kernel);
                        Cv2.Max(historyWarped, minNeighbor, historyWarped);
                        Cv2.Min(historyWarped, maxNeighbor, historyWarped);

                        // 5. BLEND FACTOR CALCULATION (Balanced)
                        // Base Alpha = MinBlend
                        UMat alphaMat = new UMat(new Size(width, height), MatType.CV_32F, new Scalar(MinBlend));

                        // Velocity-based boost
                        UMat flowMagnitude = new UMat();
                        UMat flowX = new UMat();
                        UMat flowY = new UMat();
                        UMat velocityAlpha = new UMat();

                        Cv2.ExtractChannel(flowFwd, flowX, 0);
                        Cv2.ExtractChannel(flowFwd, flowY, 1);
                        Cv2.Magnitude(flowX, flowY, flowMagnitude);

                        // Normalize velocity into [0..1] over [MinVelocityThreshold..MaxVelocityThreshold]
                        Cv2.Subtract(flowMagnitude, new Scalar(MinVelocityThreshold), velocityAlpha);

                        double vRange = MaxVelocityThreshold - MinVelocityThreshold;
                        if (vRange < 1.0) vRange = 1.0;

                        Cv2.Multiply(velocityAlpha, new Scalar(1.0 / vRange), velocityAlpha);

                        // Clamp velocityAlpha to [0,1]
                        Cv2.Threshold(velocityAlpha, velocityAlpha, 1.0, 1.0, ThresholdTypes.Trunc);
                        Cv2.Threshold(velocityAlpha, velocityAlpha, 0.0, 0.0, ThresholdTypes.Tozero);

                        UMat velocityBoost = new UMat();
                        Cv2.Multiply(velocityAlpha, new Scalar(MaxBlend - MinBlend), velocityBoost);
                        Cv2.Add(alphaMat, velocityBoost, alphaMat);

                        // --- 5b. STATIC-REGION BIAS ---
                        // Pixels with extremely low motion get slightly lower alpha
                        // (more history) to reduce shimmer on static details.
                        UMat staticMask = new UMat();
                        Cv2.Threshold(flowMagnitude, staticMask, StaticVelocityThreshold, 1.0, ThresholdTypes.BinaryInv);
                        // staticMask: 1 where flowMagnitude <= StaticVelocityThreshold, else 0

                        UMat staticScale = new UMat(staticMask.Size(), MatType.CV_32F);
                        // staticScale = StaticAlphaScale where static, 1.0 elsewhere
                        Cv2.Multiply(staticMask, new Scalar(StaticAlphaScale - 1.0), staticScale);
                        Cv2.Add(staticScale, new Scalar(1.0), staticScale);

                        // alphaMat *= staticScale
                        Cv2.Multiply(alphaMat, staticScale, alphaMat);

                        // 6. MOTION MASK: enforce a higher minimum alpha in motion regions
                        // flowMagnitude > MotionThreshold ⇒ motionMask = 1.0
                        UMat motionMask = new UMat();
                        Cv2.Threshold(flowMagnitude, motionMask, MotionThreshold, 1.0, ThresholdTypes.Binary);

                        // Expand motion region slightly
                        Cv2.Dilate(motionMask, motionMask, dilationKernel);

                        // motionAlphaMin = MotionMinAlpha where motionMask=1, otherwise 0
                        UMat motionAlphaMin = new UMat(motionMask.Size(), MatType.CV_32F);
                        Cv2.Multiply(motionMask, new Scalar(MotionMinAlpha), motionAlphaMin);

                        // alphaMat = max(alphaMat, motionAlphaMin)
                        Cv2.Max(alphaMat, motionAlphaMin, alphaMat);

                        // 7. APPLY OCCLUSION MASK
                        // Wherever consistency failed (Mask=1.0), force Alpha to 1.0 (pure current frame).
                        Cv2.Max(alphaMat, gpuConsistencyMask, alphaMat);

                        // 7b. HIGH-VELOCITY SAFETY
                        // Wherever motion is extremely large (gpuBigMask=1.0), also force Alpha to 1.0.
                        // This prevents "stretching" warpy history in very fast/untrusted motion.
                        Cv2.Max(alphaMat, gpuBigMask, alphaMat);

                        // 8. ACCUMULATE
                        UMat alpha3 = new UMat();
                        UMat invAlpha3 = new UMat();
                        UMat part1 = new UMat();
                        UMat part2 = new UMat();

                        Cv2.CvtColor(alphaMat, alpha3, ColorConversionCodes.GRAY2BGR);
                        Cv2.Subtract(new Scalar(1.0, 1.0, 1.0), alpha3, invAlpha3);

                        Cv2.Multiply(currentFloat, alpha3, part1);
                        Cv2.Multiply(historyWarped, invAlpha3, part2);
                        Cv2.Add(part1, part2, accumulationBuffer);
                    }

                    // 9. OUTPUT
                    currentGray.CopyTo(prevGray);
                    accumulationBuffer.ConvertTo(finalOutput, MatType.CV_8UC3);

                    if (SharpenAmount > 0.0)
                    {
                        Cv2.Filter2D(finalOutput, finalOutput, -1, sharpenKernel);
                    }

                    writer.Write(finalOutput);

                    frameIndex++;
                    if (frameIndex % 10 == 0)
                        Console.Write($"\rProcessing Frame: {frameIndex}/{totalFrames} ({(double)frameIndex / totalFrames:P1})");
                }
            }
        }
    }

    static void CreateRemapGrid(int w, int h, UMat mapX, UMat mapY)
    {
        using (Mat mX = new Mat(h, w, MatType.CV_32F))
        using (Mat mY = new Mat(h, w, MatType.CV_32F))
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    mX.Set(y, x, (float)x);
                    mY.Set(y, x, (float)y);
                }
            }
            mX.CopyTo(mapX);
            mY.CopyTo(mapY);
        }
    }

    static void UpdateRemapMaps(UMat mapX, UMat mapY, UMat flow)
    {
        using (Mat mFlow = flow.GetMat(AccessFlag.READ))
        using (Mat mMapX = mapX.GetMat(AccessFlag.READ))
        using (Mat mMapY = mapY.GetMat(AccessFlag.READ))
        {
            var idxFlow = mFlow.GetGenericIndexer<Vec2f>();
            var idxX = mMapX.GetGenericIndexer<float>();
            var idxY = mMapY.GetGenericIndexer<float>();

            int w = mFlow.Cols;
            int h = mFlow.Rows;

            Parallel.For(0, h, y =>
            {
                for (int x = 0; x < w; x++)
                {
                    Vec2f f = idxFlow[y, x];

                    // Note: we move back along the flow to sample history.
                    idxX[y, x] = x - f.Item0;
                    idxY[y, x] = y - f.Item1;
                }
            });
        }
    }

    static bool MuxAudio(string originalVideo, string processedVideo, string finalOutput)
    {
        string args = $"-y -i \"{processedVideo}\" -i \"{originalVideo}\" -c:v copy -c:a copy -map 0:v -map 1:a -shortest \"{finalOutput}\"";

        try
        {
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = "ffmpeg",
                Arguments = args,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using (Process p = new Process { StartInfo = psi })
            {
                p.OutputDataReceived += (sender, e) =>
                {
                    if (e.Data != null) Console.WriteLine($"[FFmpeg Info] {e.Data}");
                };
                p.ErrorDataReceived += (sender, e) =>
                {
                    if (e.Data != null) Console.WriteLine($"[FFmpeg] {e.Data}");
                };

                p.Start();
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                p.WaitForExit();
                return p.ExitCode == 0;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FFmpeg Error: {ex.Message}");
            return false;
        }
    }
}
