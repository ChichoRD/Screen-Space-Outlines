using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class ScreenSpaceOutlinesRenderPassFeature : ScriptableRendererFeature
{
    class ScreenSpaceOutlinesRenderPass : ScriptableRenderPass
    {
        private const string SCREEN_SPACE_OUTLINES_SHADER_PATH = "Hidden/Screen Space Outlines";
        private Material _screenSpaceOutlinesMaterial;
        private static readonly string s_PassName = nameof(ScreenSpaceOutlinesRenderPass);
        private static readonly ProfilingSampler s_profilingSampler = new ProfilingSampler(s_PassName);
        private readonly ScreenSpaceOutlinesRenderPassSettings _settings;

        private RenderTargetIdentifier _temporaryRenderTarget0;
        private RenderTargetIdentifier _temporaryRenderTarget1;

        private static readonly int s_temporaryRenderTarget0ID = Shader.PropertyToID(s_PassName + nameof(_temporaryRenderTarget0));
        private static readonly int s_temporaryRenderTarget1ID = Shader.PropertyToID(s_PassName + nameof(_temporaryRenderTarget1));

        private static readonly int s_JfaExtentsID = Shader.PropertyToID("_JfaExtents");

        private static readonly int s_ColorThresholdFactorID = Shader.PropertyToID("_ColorThresholdFactor");
        private static readonly int s_NormalThresholdFactorID = Shader.PropertyToID("_NormalThresholdFactor");
        private static readonly int s_GeometryThresholdFactorID = Shader.PropertyToID("_GeometryThresholdFactor");
        private static readonly int s_OutlineThresholdID = Shader.PropertyToID("_OutlineThreshold");

        private static readonly int s_KawaseBlurStepRadiusID = Shader.PropertyToID("_KawaseBlurStepRadius");

        private static readonly int s_OutlineColorID = Shader.PropertyToID("_OutlineColor");
        private static readonly int s_OutlineIntensityID = Shader.PropertyToID("_OutlineIntensity");
        private static readonly int s_OutlineDrawTightnessID = Shader.PropertyToID("_OutlineDrawTightness");
        private static readonly int s_OutlineDrawThresholdID = Shader.PropertyToID("_OutlineDrawThreshold");

        private static readonly int s_ConvexityThresholdID = Shader.PropertyToID("_ConvexityThreshold");
        private static readonly int s_ConcavityThresholdID = Shader.PropertyToID("_ConcavityThreshold");
        private static readonly int s_ConvexityOverlayColorID = Shader.PropertyToID("_ConvexityOverlayColor");
        private static readonly int s_ConcavityOverlayColorID = Shader.PropertyToID("_ConcavityOverlayColor");

        private static readonly int s_DistanceFadeFactorID = Shader.PropertyToID("_DistanceFadeFactor");

        public ScreenSpaceOutlinesRenderPass(ScreenSpaceOutlinesRenderPassSettings settings)
        {
            _settings = settings;
        }

        // This method is called before executing the render pass.
        // It can be used to configure render targets and their clear state. Also to create temporary render target textures.
        // When empty this render pass will render to the active camera render target.
        // You should never call CommandBuffer.SetRenderTarget. Instead call <c>ConfigureTarget</c> and <c>ConfigureClear</c>.
        // The render pipeline will ensure target setup and clearing happens in a performant manner.
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            base.OnCameraSetup(cmd, ref renderingData);
            RenderTextureDescriptor descriptor = renderingData.cameraData.cameraTargetDescriptor;
            descriptor.colorFormat = _settings.FromTextureEncoding();
            descriptor.depthBufferBits = 0;

            cmd.GetTemporaryRT(s_temporaryRenderTarget0ID, descriptor, FilterMode.Bilinear);
            cmd.GetTemporaryRT(s_temporaryRenderTarget1ID, descriptor, FilterMode.Bilinear);

            _temporaryRenderTarget0 = new RenderTargetIdentifier(s_temporaryRenderTarget0ID);
            _temporaryRenderTarget1 = new RenderTargetIdentifier(s_temporaryRenderTarget1ID);

            _screenSpaceOutlinesMaterial = _screenSpaceOutlinesMaterial == null
                                           ? CoreUtils.CreateEngineMaterial(SCREEN_SPACE_OUTLINES_SHADER_PATH)
                                           : _screenSpaceOutlinesMaterial;

            _screenSpaceOutlinesMaterial.SetFloat(s_ColorThresholdFactorID, _settings.ColorThresholdFactor);
            _screenSpaceOutlinesMaterial.SetFloat(s_NormalThresholdFactorID, _settings.NormalThresholdFactor);
            _screenSpaceOutlinesMaterial.SetFloat(s_GeometryThresholdFactorID, _settings.GeometryThresholdFactor);
            _screenSpaceOutlinesMaterial.SetFloat(s_OutlineThresholdID, _settings.OutlineThreshold);

            _screenSpaceOutlinesMaterial.SetColor(s_OutlineColorID, _settings.OutlineColor);
            _screenSpaceOutlinesMaterial.SetFloat(s_OutlineIntensityID, _settings.OutlineIntensity);
            _screenSpaceOutlinesMaterial.SetFloat(s_OutlineDrawTightnessID, _settings.OutlineDrawTightness);
            _screenSpaceOutlinesMaterial.SetFloat(s_OutlineDrawThresholdID, _settings.OutlineDrawThreshold);

            _screenSpaceOutlinesMaterial.SetFloat(s_ConvexityThresholdID, _settings.ConvexityThreshold);
            _screenSpaceOutlinesMaterial.SetFloat(s_ConcavityThresholdID, _settings.ConcavityThreshold);
            _screenSpaceOutlinesMaterial.SetColor(s_ConvexityOverlayColorID, _settings.ConvexityOverlayColor);
            _screenSpaceOutlinesMaterial.SetColor(s_ConcavityOverlayColorID, _settings.ConcavityOverlayColor);

            _screenSpaceOutlinesMaterial.SetFloat(s_DistanceFadeFactorID, _settings.DistanceFadeFactor);

            ConfigureInput(ScriptableRenderPassInput.Color
                           | ScriptableRenderPassInput.Depth
                           | ScriptableRenderPassInput.Normal);

            ConfigureTarget(_temporaryRenderTarget0);
            ConfigureClear(ClearFlag.All, Color.black);
        }

        // Here you can implement the rendering logic.
        // Use <c>ScriptableRenderContext</c> to issue drawing commands or execute command buffers
        // https://docs.unity3d.com/ScriptReference/Rendering.ScriptableRenderContext.html
        // You don't have to call ScriptableRenderContext.submit, the render pipeline will call it at specific points in the pipeline.
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get();
            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();

            using (new ProfilingScope(cmd, s_profilingSampler))
            {
                RenderTargetIdentifier cameraColorTarget = renderingData.cameraData.renderer.cameraColorTarget;
                RenderTextureDescriptor cameraTargetDescriptor = renderingData.cameraData.cameraTargetDescriptor;

                cmd.Blit(cameraColorTarget, _temporaryRenderTarget0, _screenSpaceOutlinesMaterial, 0);
                int minExtent = Mathf.Min(cameraTargetDescriptor.width, cameraTargetDescriptor.height) >> (_settings.RangeHalving + 1);

                //JFA
                for (int extents = minExtent; extents > 0; extents /= 2)
                {
                    cmd.SetGlobalVector(s_JfaExtentsID, new Vector2(extents, extents));
                    cmd.Blit(_temporaryRenderTarget0, _temporaryRenderTarget1, _screenSpaceOutlinesMaterial, 1);
                    cmd.Blit(_temporaryRenderTarget1, _temporaryRenderTarget0);
                }

                //JFA + N
                for (int extents = 1 << (_settings.ExtraIterations - 1); extents > 0; extents /= 2)
                {
                    cmd.SetGlobalVector(s_JfaExtentsID, new Vector2(extents, extents));
                    cmd.Blit(_temporaryRenderTarget0, _temporaryRenderTarget1, _screenSpaceOutlinesMaterial, 1);
                    cmd.Blit(_temporaryRenderTarget1, _temporaryRenderTarget0);
                }

                cmd.Blit(_temporaryRenderTarget0, cameraColorTarget, _screenSpaceOutlinesMaterial, 2);
            }

            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);
        }

        // Cleanup any allocated resources that were created during the execution of this render pass.
        public override void OnCameraCleanup(CommandBuffer cmd)
        {
            base.OnCameraCleanup(cmd);

            cmd ??= CommandBufferPool.Get();
            cmd.ReleaseTemporaryRT(s_temporaryRenderTarget0ID);
            cmd.ReleaseTemporaryRT(s_temporaryRenderTarget1ID);
        }
    }

    [Serializable]
    private struct ScreenSpaceOutlinesRenderPassSettings
    {
        public static readonly ScreenSpaceOutlinesRenderPassSettings s_Default = new ScreenSpaceOutlinesRenderPassSettings()
        {
            ColorThresholdFactor = 0.5f,
            NormalThresholdFactor = 0.75f,
            GeometryThresholdFactor = 0.1f,
            OutlineThreshold = 0.5f,

            OutlineColor = new Color(0.15f, 0.15f, 0.15f),
            OutlineIntensity = 3.0f,
            OutlineDrawTightness = 0.0025f,
            OutlineDrawThreshold = 0.9975f,

            ConvexityThreshold = 1.0f,
            ConcavityThreshold = 0.0f,
            ConvexityOverlayColor = Color.gray,
            ConcavityOverlayColor = Color.gray,

            DistanceFadeFactor = 0.05f,
            TextureEncodingOptions = TextureEncoding.ARGB2101010,
            RenderPassEvent = RenderPassEvent.AfterRenderingOpaques
        };

        [field: SerializeField]
        [field: Range(0.0f, 1.0f)]
        public float ColorThresholdFactor { get; private set; }
        [field: SerializeField]
        [field: Range(0.0f, 1.0f)]
        public float NormalThresholdFactor { get; private set; }
        [field: SerializeField]
        [field: Range(0.0f, 1.0f)]
        public float GeometryThresholdFactor { get; private set; }

        [field: SerializeField]
        [field: Range(0.0f, 1.0f)]
        public float OutlineThreshold { get; private set; }

        [field: Space]
        [field: SerializeField]
        [field: ColorUsage(true, true)]
        public Color OutlineColor { get; private set; }
        [field: SerializeField]
        [field: Min(0.0f)]
        public float OutlineIntensity { get; private set; }

        [field: SerializeField]
        [field: Range(0.0f, 1.0f)]
        public float OutlineDrawTightness { get; private set; }
        [field: SerializeField]
        [field: Range(0.0f, 1.0f)]
        public float OutlineDrawThreshold { get; private set; }

        [field: Space]
        [field: SerializeField]
        [field: Range(0.0f, 1.0f)]
        public float ConvexityThreshold { get; private set; }
        [field: SerializeField]
        [field: Range(0.0f, 1.0f)]
        public float ConcavityThreshold { get; private set; }
        [field: SerializeField]
        [field: ColorUsage(true, true)]
        public Color ConvexityOverlayColor { get; private set; }
        [field: SerializeField]
        [field: ColorUsage(true, true)]
        public Color ConcavityOverlayColor { get; private set; }

        [field: Space]
        [field: SerializeField]
        [field: Min(0.0f)]
        public float DistanceFadeFactor { get; private set; }
        [field: SerializeField]
        public TextureEncoding TextureEncodingOptions { get; private set; }
        [field: SerializeField]
        [field: Min(0)]
        public int RangeHalving { get; private set; }
        [field: SerializeField]
        [field: Range(0, 2)]
        public int ExtraIterations { get; private set; }
        [field: SerializeField]
        public RenderPassEvent RenderPassEvent { get; private set; }

        public readonly RenderTextureFormat FromTextureEncoding() => TextureEncodingOptions switch
        {
            TextureEncoding.ARGB2101010 => RenderTextureFormat.ARGB2101010,
            TextureEncoding.ARGBHalf => RenderTextureFormat.ARGBHalf,
            TextureEncoding.ARGBFloat => RenderTextureFormat.ARGBFloat,
            _ => throw new ArgumentOutOfRangeException()
        };

        [Serializable]
        public enum TextureEncoding
        {
            ARGB2101010,
            ARGBHalf,
            ARGBFloat
        }
    }

    private ScreenSpaceOutlinesRenderPass _screenSpaceOutlinesRenderPass;
    [SerializeField]
    private ScreenSpaceOutlinesRenderPassSettings _settings = ScreenSpaceOutlinesRenderPassSettings.s_Default;

    /// <inheritdoc/>
    public override void Create()
    {
        _screenSpaceOutlinesRenderPass = new ScreenSpaceOutlinesRenderPass(_settings);
        _screenSpaceOutlinesRenderPass.renderPassEvent = _settings.RenderPassEvent;
    }

    // Here you can inject one or multiple render passes in the renderer.
    // This method is called when setting up the renderer once per-camera.
    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(_screenSpaceOutlinesRenderPass);
    }
}


