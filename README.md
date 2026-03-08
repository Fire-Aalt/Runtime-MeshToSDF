# Runtime-MeshToSDF

Runtime-MeshToSDF turns a regular 3D mesh into a 3D Signed Distance Field (SDF) texture at runtime.
In simple terms, it creates a volume texture where each voxel stores how far it is from the mesh surface:
negative values are inside the mesh, positive values are outside.

This is useful when you want effects that react to shape and distance, such as particles colliding with a mesh
in VFX Graph, soft volumetric masking, or gameplay systems that need fast inside/outside checks.

Runtime-MeshToSDF uses GPU compute shaders and quasi-random triangle sampling, so it can generate SDFs quickly enough
for iteration and dynamic content, specifically focusing on runtime optimised API and as little overhead as possible.

## Usage Example
```csharp
using UnityEngine;
using KrasCore.MeshToSDF;

public class SdfExample : MonoBehaviour
{
    [SerializeField] private Mesh mesh;
    [SerializeField] private int sdfResolution = 64;
    [SerializeField] private uint samplesPerTriangle = 256;

    // Always cache a baker as the initialization and cache creation are expensive.
    private MeshToSDFBaker _baker;

    private void Awake()
    {
        _baker = new MeshToSDFBaker();
    }

    private void Update()
    {
        if (mesh == null) return;

        var size = mesh.bounds.size;
        var center = mesh.bounds.center;

        _baker.BakeSDF(size, center, sdfResolution, mesh, samplesPerTriangle);
        
        // If possible, use a more performant overload without a mesh: BakeSDF(Vector3 size, Vector3 center, int maxSdfResolution, NativeArray<int> indices, NativeArray<Vector3> vertices, uint samplesPerTriangle);
        // Use _baker.SdfTexture in VFX Graph or your own systems.
    }

    // Disposal is not necessary as the baker implements a destructor.
    //private void OnDestroy()
    //{
    //    _baker?.Dispose();
    //}
}
```

## Benchmark
1. Open `Demo/Demo.unity` for a working setup.
2. Set a mesh in `SDFBakersComparison`. It will be converted to SDF every frame.
3. Choose your baker to check the performance and quality of the SDF. Use frame stats and check GPU Profiler for the most accurate data.

## Performance
Benchmarked against `UnityEngine.VFX.SDF.MeshToSDFBaker` from `Visual Effects Graph`, `KrasCore.MeshToSDF.MeshToSDFBaker` is on average **2x faster**,
while also providing a no-mesh overload, requiring only vertex and index buffers (`NativeList<Vector3>` and `NativeList<int>`), which results in much smaller CPU overhead and no need for a mesh at all.

## Quality Note
`samplesPerTriangle` controls voxelization quality versus bake time.
Higher values generally produce cleaner SDFs (fewer holes/artifacts), especially on complex meshes and higher resolutions,
at the cost of slightly more GPU work (going from 64 to 256 `samplesPerTriangle` costs about 1-3% overhead)

### Special Thanks
To the original implementation: https://github.com/aman-tiwari/MeshToSDF