using System;
using Unity.Collections;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Rendering;

namespace KrasCore.MeshToSDF
{
    public class MeshToSDFBaker : IDisposable
    {
        private static class ShaderProperties
        {
            public static readonly int InsideMask = Shader.PropertyToID("InsideMask");
            public static readonly int Voxels = Shader.PropertyToID("Voxels");
            public static readonly int VertexBuffer = Shader.PropertyToID("VertexBuffer");
            public static readonly int IndexBuffer = Shader.PropertyToID("IndexBuffer");
            public static readonly int VoxelSize = Shader.PropertyToID("voxelSize");
            public static readonly int VoxelScale = Shader.PropertyToID("voxelScale");
            public static readonly int VoxelOrigin = Shader.PropertyToID("voxelOrigin");
            public static readonly int NumSamples = Shader.PropertyToID("numSamples");
            public static readonly int Tris = Shader.PropertyToID("tris");
            public static readonly int OutputSdf = Shader.PropertyToID("OutputSdf");
            public static readonly int SamplingOffset = Shader.PropertyToID("samplingOffset");
            public static readonly int VoteThreshold = Shader.PropertyToID("insideVoteThreshold");
            public static readonly int DistanceNormalization = Shader.PropertyToID("distanceNormalization");
            public static readonly int DispatchSize = Shader.PropertyToID("dispatchSize");
        }
        
        private class Kernels
        {
            public int Jfa;
            public int Preprocess;
            public int Postprocess;
            public int MtV;
            public int Zero;
            public int ClearInsideMask;
            public int ClassifyInsideX;
            public int ClassifyInsideY;
            public int ClassifyInsideZ;
        }
        
        private static ProfilerMarker _extractMeshDataMarker = new("MeshToSDF.ExtractMeshData");
        private static ProfilerMarker _bakeSdfMarker = new("MeshToSDF.BakeSDF");
        private const string VoxelisationGpuMarker = "MeshToSDF.VoxelisationGPU";
        private const string VoxelisationGpuMarkerZero = "MeshToSDF.VoxelisationGPU.Zero";
        private const string VoxelisationGpuMarkerMeshToVoxel = "MeshToSDF.VoxelisationGPU.MeshToVoxel";
        private const string VoxelisationGpuMarkerClearInsideMask = "MeshToSDF.VoxelisationGPU.ClearInsideMask";
        private const string VoxelisationGpuMarkerClassifyInside = "MeshToSDF.VoxelisationGPU.ClassifyInside";
        private const string VoxelisationGpuMarkerClassifyInsideX = "MeshToSDF.VoxelisationGPU.ClassifyInside.X";
        private const string VoxelisationGpuMarkerClassifyInsideY = "MeshToSDF.VoxelisationGPU.ClassifyInside.Y";
        private const string VoxelisationGpuMarkerClassifyInsideZ = "MeshToSDF.VoxelisationGPU.ClassifyInside.Z";
        private const string SdfJumpFillGpuMarker = "MeshToSDF.SDFJumpFillGPU";
        private const string SdfJumpFillGpuMarkerPreprocess = "MeshToSDF.SDFJumpFillGPU.Preprocess";
        private const string SdfJumpFillGpuMarkerJfa = "MeshToSDF.SDFJumpFillGPU.JFA";
        private const string SdfJumpFillGpuMarkerJfaStep = "MeshToSDF.SDFJumpFillGPU.JFA.Step";
        private const string SdfJumpFillGpuMarkerPostprocess = "MeshToSDF.SDFJumpFillGPU.Postprocess";
        const int InsideVoteThreshold = 3;     
        
        public RenderTexture SdfTexture => _sdfTexture;
        
        private Vector3 _size;
        private Vector3 _center;
        private uint _samplesPerTriangle;

        private readonly string _sdfTextureName;
        private readonly Kernels _kernels;
        private ComputeShader _jfaShader;
        private ComputeShader _mtvShader;
        
        private readonly ComputeBuffer[] _cachedBuffers = new ComputeBuffer[2];
        private readonly float[] _cachedFloat3 = new float[3];
        private readonly int[] _cachedInt3 = new int[3];
        private NativeList<int> _cachedIndices = new(Allocator.Persistent);
        private NativeList<Vector3> _cachedVertices = new(Allocator.Persistent);
        
        private readonly CommandBuffer _gpuCommandBuffer;
        private RenderTexture _voxelTexture;
        private RenderTexture _insideMaskTexture;
        private RenderTexture _sdfTexture;
        
        public MeshToSDFBaker(string sdfTextureName = "SDFTexture")
        {
            _sdfTextureName = sdfTextureName;

            LoadComputeShaders();
            _kernels = new Kernels
            {
                Jfa = _jfaShader.FindKernel("JFA"),
                Preprocess = _jfaShader.FindKernel("Preprocess"),
                Postprocess = _jfaShader.FindKernel("Postprocess"),
                MtV = _mtvShader.FindKernel("MeshToVoxel"),
                Zero = _mtvShader.FindKernel("Zero"),
                ClearInsideMask = _mtvShader.FindKernel("ClearInsideMask"),
                ClassifyInsideX = _mtvShader.FindKernel("ClassifyInsideX"),
                ClassifyInsideY = _mtvShader.FindKernel("ClassifyInsideY"),
                ClassifyInsideZ = _mtvShader.FindKernel("ClassifyInsideZ"),
            };
            
            _gpuCommandBuffer = new CommandBuffer
            {
                name = "MeshToSDF.GPUProfiling"
            };
        }
        
        /// <summary>
        /// Bakes a signed distance field from a Unity mesh by extracting its vertex/index data.
        /// </summary>
        /// <param name="size">Bounds size of the SDF volume.</param>
        /// <param name="center">Center of the SDF volume.</param>
        /// <param name="maxSdfResolution">Target resolution along the largest bounds axis.</param>
        /// <param name="mesh">Source mesh to voxelize.</param>
        /// <param name="samplesPerTriangle">Number of sampling points used per triangle during voxelization. It is very cheap to increase this parameter but can smooth out the SDF drastically.</param>
        public void BakeSDF(Vector3 size, Vector3 center, int maxSdfResolution, Mesh mesh,
            uint samplesPerTriangle = 128) 
        {
            ExtractMeshData(mesh, out var indices, out var vertices);
            BakeSDF(size, center, maxSdfResolution, indices, vertices, samplesPerTriangle);
        }
        
        /// <summary>
        /// Bakes a signed distance field from raw mesh index and vertex buffers.
        /// </summary>
        /// <param name="size">Bounds size of the SDF volume.</param>
        /// <param name="center">Center of the SDF volume.</param>
        /// <param name="maxSdfResolution">Target resolution along the largest bounds axis.</param>
        /// <param name="indices">Triangle index buffer (3 indices per triangle).</param>
        /// <param name="vertices">Vertex buffer referenced by <paramref name="indices"/>.</param>
        /// <param name="samplesPerTriangle">Number of sampling points used per triangle during voxelization. It is very cheap to increase this parameter but can smooth out the SDF drastically.</param>
        public void BakeSDF(Vector3 size, Vector3 center, int maxSdfResolution, NativeArray<int> indices, NativeArray<Vector3> vertices,
            uint samplesPerTriangle = 128) 
        {
            _bakeSdfMarker.Begin();
            _size = size;
            _center = center;
            _samplesPerTriangle = samplesPerTriangle;
            
            BeginDispatchRecording();

            _voxelTexture = MeshToVoxel(maxSdfResolution, indices, vertices, _voxelTexture);
            _sdfTexture = EnsureSdfOutputTexture(_voxelTexture, _sdfTexture);
            FloodFillToSDF(maxSdfResolution, _voxelTexture, _insideMaskTexture, _sdfTexture);

            EndDispatchRecording();
            _bakeSdfMarker.End();
        }

        private void ExtractMeshData(Mesh mesh, out NativeArray<int> indices, out NativeArray<Vector3> vertices) 
        {
            _extractMeshDataMarker.Begin();
            var meshDataArray = Mesh.AcquireReadOnlyMeshData(mesh);
            var meshData = meshDataArray[0];
            var indexCount = (int)mesh.GetIndexCount(0);
            var vertexCount = mesh.vertexCount;
            
            _cachedIndices.ResizeUninitialized(indexCount);
            meshData.GetIndices(_cachedIndices.AsArray(), 0);
            indices = _cachedIndices.AsArray();
            
            _cachedVertices.ResizeUninitialized(vertexCount);
            meshData.GetVertices(_cachedVertices.AsArray());
            vertices = _cachedVertices.AsArray();
            
            meshDataArray.Dispose();
            _extractMeshDataMarker.End();
        }

        private RenderTexture MeshToVoxel(int voxelResolution, NativeArray<int> indices, NativeArray<Vector3> vertices,
            RenderTexture voxels)
        {
            var triangleCount = indices.Length / 3;
            var voxelScale = VoxelScaleFromBounds(voxelResolution);
            var voxelSize = VoxelDimensions(voxelScale);
            var voxelOrigin = _center - _size / 2f;
            
            // Indices
            var indicesBuffer = GetCachedComputeBuffer(indices.Length, sizeof(uint), 0);
            indicesBuffer.SetData(indices);
            
            // Vertices
            var vertexBuffer = GetCachedComputeBuffer(vertices.Length, sizeof(float) * 3, 1);
            vertexBuffer.SetData(vertices);
            
            _mtvShader.SetBuffer(_kernels.MtV, ShaderProperties.IndexBuffer, indicesBuffer);
            _mtvShader.SetBuffer(_kernels.MtV, ShaderProperties.VertexBuffer, vertexBuffer);
            _mtvShader.SetInt(ShaderProperties.Tris, triangleCount);
            _mtvShader.SetInt(ShaderProperties.NumSamples, (int)_samplesPerTriangle);
            _cachedFloat3[0] = voxelOrigin.x; _cachedFloat3[1] = voxelOrigin.y; _cachedFloat3[2] = voxelOrigin.z;
            _mtvShader.SetFloats(ShaderProperties.VoxelOrigin, _cachedFloat3);
            _mtvShader.SetFloat(ShaderProperties.VoxelScale, voxelScale);
            _cachedInt3[0] = voxelSize.x; _cachedInt3[1] = voxelSize.y; _cachedInt3[2] = voxelSize.z;
            _mtvShader.SetInts(ShaderProperties.VoxelSize, _cachedInt3);

            if (voxels == null || voxels.width != voxelSize.x || voxels.height != voxelSize.y || voxels.volumeDepth != voxelSize.z) 
            {
                if (voxels != null) voxels.Release();
                voxels = new RenderTexture(voxelSize.x, voxelSize.y, 0, RenderTextureFormat.ARGBHalf)
                {
                    dimension = TextureDimension.Tex3D,
                    enableRandomWrite = true,
                    useMipMap = false,
                    volumeDepth = voxelSize.z
                };
                voxels.Create();
            }
            if (_insideMaskTexture == null || _insideMaskTexture.width != voxelSize.x || _insideMaskTexture.height != voxelSize.y || _insideMaskTexture.volumeDepth != voxelSize.z) 
            {
                if (_insideMaskTexture != null) _insideMaskTexture.Release();
                _insideMaskTexture = new RenderTexture(voxelSize.x, voxelSize.y, 0, RenderTextureFormat.RHalf)
                {
                    dimension = TextureDimension.Tex3D,
                    enableRandomWrite = true,
                    useMipMap = false,
                    volumeDepth = voxelSize.z
                };
                _insideMaskTexture.Create();
            }

            BeginGpuMarker(VoxelisationGpuMarker);
            _mtvShader.SetBuffer(_kernels.Zero, ShaderProperties.IndexBuffer, indicesBuffer);
            _mtvShader.SetBuffer(_kernels.Zero, ShaderProperties.VertexBuffer, vertexBuffer);
            _mtvShader.SetTexture(_kernels.Zero, ShaderProperties.Voxels, voxels);
            DispatchComputeWithGpuMarker(_mtvShader, _kernels.Zero, VoxelisationGpuMarkerZero,
                NumGroups(voxelSize.x, 8), NumGroups(voxelSize.y, 8), NumGroups(voxelSize.z, 8));

            _mtvShader.SetTexture(_kernels.MtV, ShaderProperties.Voxels, voxels);
            DispatchComputeWithGpuMarker(_mtvShader, _kernels.MtV, VoxelisationGpuMarkerMeshToVoxel, NumGroups(triangleCount, 512), 1, 1);

            _mtvShader.SetTexture(_kernels.ClearInsideMask, ShaderProperties.InsideMask, _insideMaskTexture);
            DispatchComputeWithGpuMarker(_mtvShader, _kernels.ClearInsideMask, VoxelisationGpuMarkerClearInsideMask,
                NumGroups(voxelSize.x, 8), NumGroups(voxelSize.y, 8), NumGroups(voxelSize.z, 8));

            BeginGpuMarker(VoxelisationGpuMarkerClassifyInside);
            _mtvShader.SetTexture(_kernels.ClassifyInsideX, ShaderProperties.Voxels, voxels);
            _mtvShader.SetTexture(_kernels.ClassifyInsideX, ShaderProperties.InsideMask, _insideMaskTexture);
            DispatchComputeWithGpuMarker(_mtvShader, _kernels.ClassifyInsideX, VoxelisationGpuMarkerClassifyInsideX,
                1, NumGroups(voxelSize.y, 8), NumGroups(voxelSize.z, 8));
            
            _mtvShader.SetTexture(_kernels.ClassifyInsideY, ShaderProperties.Voxels, voxels);
            _mtvShader.SetTexture(_kernels.ClassifyInsideY, ShaderProperties.InsideMask, _insideMaskTexture);
            DispatchComputeWithGpuMarker(_mtvShader, _kernels.ClassifyInsideY, VoxelisationGpuMarkerClassifyInsideY,
                NumGroups(voxelSize.x, 8), 1, NumGroups(voxelSize.z, 8));
            
            _mtvShader.SetTexture(_kernels.ClassifyInsideZ, ShaderProperties.Voxels, voxels);
            _mtvShader.SetTexture(_kernels.ClassifyInsideZ, ShaderProperties.InsideMask, _insideMaskTexture);
            DispatchComputeWithGpuMarker(_mtvShader, _kernels.ClassifyInsideZ, VoxelisationGpuMarkerClassifyInsideZ,
                NumGroups(voxelSize.x, 8), NumGroups(voxelSize.y, 8), 1);
            EndGpuMarker(VoxelisationGpuMarkerClassifyInside);
            EndGpuMarker(VoxelisationGpuMarker);

            return voxels;
        }

        private void FloodFillToSDF(int sdfResolution, RenderTexture voxels, RenderTexture insideMask, RenderTexture outputSdf)
        {
            _cachedInt3[0] = voxels.width; _cachedInt3[1] = voxels.height; _cachedInt3[2] = voxels.volumeDepth;
            _jfaShader.SetInts(ShaderProperties.DispatchSize, _cachedInt3);
            _jfaShader.SetFloat(ShaderProperties.DistanceNormalization, Mathf.Max(1, sdfResolution));
            _jfaShader.SetInt(ShaderProperties.VoteThreshold, InsideVoteThreshold);

            BeginGpuMarker(SdfJumpFillGpuMarker);
            _jfaShader.SetTexture(_kernels.Preprocess, ShaderProperties.Voxels, voxels);
            DispatchComputeWithGpuMarker(_jfaShader, _kernels.Preprocess, SdfJumpFillGpuMarkerPreprocess,
                NumGroups(voxels.width, 8), NumGroups(voxels.height, 8), NumGroups(voxels.volumeDepth, 8));

            _jfaShader.SetTexture(_kernels.Jfa, ShaderProperties.Voxels, voxels);
            BeginGpuMarker(SdfJumpFillGpuMarkerJfa);
            int maxSide = Mathf.Max(voxels.width, Mathf.Max(voxels.height, voxels.volumeDepth));
            for (int offset = Mathf.NextPowerOfTwo(maxSide) / 2; offset >= 1; offset /= 2) {
                _gpuCommandBuffer.SetComputeIntParam(_jfaShader, ShaderProperties.SamplingOffset, offset);
                DispatchComputeWithGpuMarker(_jfaShader, _kernels.Jfa, SdfJumpFillGpuMarkerJfaStep,
                    NumGroups(voxels.width, 8), NumGroups(voxels.height, 8), NumGroups(voxels.volumeDepth, 8));
            }
            EndGpuMarker(SdfJumpFillGpuMarkerJfa);

            _jfaShader.SetTexture(_kernels.Postprocess, ShaderProperties.Voxels, voxels);
            _jfaShader.SetTexture(_kernels.Postprocess, ShaderProperties.InsideMask, insideMask);
            _jfaShader.SetTexture(_kernels.Postprocess, ShaderProperties.OutputSdf, outputSdf);

            DispatchComputeWithGpuMarker(_jfaShader, _kernels.Postprocess, SdfJumpFillGpuMarkerPostprocess,
                NumGroups(voxels.width, 8), NumGroups(voxels.height, 8), NumGroups(voxels.volumeDepth, 8));
            EndGpuMarker(SdfJumpFillGpuMarker);
        }
        
        private RenderTexture EnsureSdfOutputTexture(RenderTexture voxels, RenderTexture outputSdf)
        {
            if (outputSdf == null || outputSdf.width != voxels.width || outputSdf.height != voxels.height || outputSdf.volumeDepth != voxels.volumeDepth) 
            {
                if (outputSdf != null) outputSdf.Release();
                outputSdf = new RenderTexture(voxels.width, voxels.height, 0, RenderTextureFormat.RHalf)
                {
                    name = _sdfTextureName,
                    dimension = TextureDimension.Tex3D,
                    enableRandomWrite = true,
                    useMipMap = false,
                    volumeDepth = voxels.volumeDepth
                };
                outputSdf.Create();
            }

            return outputSdf;
        }

        private Vector3Int VoxelDimensions(float voxelScale)
        {
            var x = Mathf.Max(1, Mathf.CeilToInt(Mathf.Abs(_size.x) * voxelScale));
            var y = Mathf.Max(1, Mathf.CeilToInt(Mathf.Abs(_size.y) * voxelScale));
            var z = Mathf.Max(1, Mathf.CeilToInt(Mathf.Abs(_size.z) * voxelScale));
            return new Vector3Int(x, y, z);
        }

        private float VoxelScaleFromBounds(int voxelResolution) 
        {
            var maxSize = Mathf.Max(_size.x, Mathf.Max(_size.y, _size.z));
            if (maxSize <= 0.0f) return 1.0f;
            return voxelResolution / maxSize;
        }

        private void BeginDispatchRecording()
        {
            _gpuCommandBuffer.Clear();
        }

        private void EndDispatchRecording()
        {
            if (_gpuCommandBuffer == null) return;
            Graphics.ExecuteCommandBuffer(_gpuCommandBuffer);
        }

        private void DispatchComputeWithGpuMarker(ComputeShader shader, int kernel, string markerName,
            int threadGroupsX, int threadGroupsY, int threadGroupsZ) 
        {
            _gpuCommandBuffer.BeginSample(markerName);
            _gpuCommandBuffer.DispatchCompute(shader, kernel, threadGroupsX, threadGroupsY, threadGroupsZ);
            _gpuCommandBuffer.EndSample(markerName);
        }

        private void BeginGpuMarker(string markerName)
        {
            _gpuCommandBuffer.BeginSample(markerName);
        }

        private void EndGpuMarker(string markerName)
        {
            _gpuCommandBuffer.EndSample(markerName);
        }

        private void LoadComputeShaders()
        {
            _jfaShader = Resources.Load<ComputeShader>("JumpFloodAssignment");
            _mtvShader = Resources.Load<ComputeShader>("MeshToVoxel");
        }
        
        private int NumGroups(int totalThreads, int groupSize) 
        {
            return (totalThreads + (groupSize - 1)) / groupSize;
        }
        
        private ComputeBuffer GetCachedComputeBuffer(int count, int stride, int cacheSlot) 
        {
            cacheSlot = cacheSlot == 0 ? 0 : 1;
            var buffer = _cachedBuffers[cacheSlot];
            if (buffer == null || buffer.stride != stride || buffer.count != count) 
            {
                buffer?.Dispose();
                buffer = new ComputeBuffer(count, stride);
                _cachedBuffers[cacheSlot] = buffer;
            }
            return buffer;
        }
        
        ~MeshToSDFBaker()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (_sdfTexture != null) _sdfTexture.Release();
            if (_voxelTexture != null) _voxelTexture.Release();
            if (_insideMaskTexture != null) _insideMaskTexture.Release();
            
            if (_cachedIndices.IsCreated) _cachedIndices.Dispose();
            if (_cachedVertices.IsCreated) _cachedVertices.Dispose();
            
            _cachedBuffers[0]?.Dispose();
            _cachedBuffers[1]?.Dispose();
            if (_gpuCommandBuffer != null) 
            {
                _gpuCommandBuffer.Release();
            }
        }
    }
}
