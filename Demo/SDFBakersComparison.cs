using UnityEngine;
using UnityEngine.VFX;

namespace KrasCore.MeshToSDF.Demo
{
    public class SDFBakersComparison : MonoBehaviour
    {
        public enum Baker
        {
            Unity,
            KrasCore
        }
        
        public bool setVFXGraphTexture = true;
        public Baker sdfBaker = Baker.KrasCore;
        
        [Tooltip("Mesh to convert to SDF. One of Mesh or Skinned Mesh Renderer must be set")]
        public Mesh mesh;

        [Tooltip("Visual effect whose property to set with the output SDF texture")]
        public VisualEffect vfxOutput;

        [Tooltip("Signed distance field resolution")]
        public int sdfResolution = 64;

        [Tooltip("SDF bounds size centered at local origin")]
        public Vector3 sizeExpands = new(0.1f, 0.1f, 0.1f);
        [Tooltip("SDF offset")]
        public Vector3 centerOffset = new(0f, 0f, 0f);
        
        [Tooltip("Number of points to sample on each triangle when voxeling. Set to 0 to auto-estimate.")]
        public uint samplesPerTriangle = 10;
        
        [Header("Read Only")]
        public RenderTexture outputSDFTexture;
        
        private MeshToSDFBaker _krasCoreBaker;
        private UnityEngine.VFX.SDF.MeshToSDFBaker _unityBaker;
        
        private void Awake()
        {
            _krasCoreBaker = new MeshToSDFBaker();
            _unityBaker = new UnityEngine.VFX.SDF.MeshToSDFBaker(default, default, 0, mesh);
        }

        private void Update()
        {
            if (mesh == null) return;
            var size = mesh.bounds.size;
            var center = mesh.bounds.center;
            
            size += sizeExpands;
            center += centerOffset;

            if (sdfBaker == Baker.Unity)
            {
                _unityBaker.Reinit(size, center, sdfResolution, mesh);
                _unityBaker.BakeSDF();
                outputSDFTexture = _unityBaker.SdfTexture;
            }
            else if (sdfBaker == Baker.KrasCore)
            {
                _krasCoreBaker.BakeSDF(size, center, sdfResolution, mesh, samplesPerTriangle);
                outputSDFTexture = _krasCoreBaker.SdfTexture;
            }
            
            if (vfxOutput && setVFXGraphTexture)
            {
                vfxOutput.SetVector3("SDFScale", size);
                vfxOutput.SetTexture("Texture3D", outputSDFTexture);
            }
        }

        private void OnDestroy()
        {
            _unityBaker.Dispose();
        }
    }
}
