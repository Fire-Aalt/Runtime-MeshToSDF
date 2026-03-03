// These methods have to be in a separate .hlsl file, as this is the only way the warning can be suppressed
#pragma warning(disable : 3557)

void FillInsideSegmentX(RWTexture3D<half> insideMask, uint y, uint z, int startX, int endX)
{
	for (int x = startX; x < endX; x++) {
		uint3 at = uint3((uint)x, y, z);
		insideMask[at] = insideMask[at] + 1.0h;
	}
}

void FillInsideSegmentY(RWTexture3D<half> insideMask, uint x, uint z, int startY, int endY)
{
	for (int y = startY; y < endY; y++) {
		uint3 at = uint3(x, (uint)y, z);
		insideMask[at] = insideMask[at] + 1.0h;
	}
}

void FillInsideSegmentZ(RWTexture3D<half> insideMask, uint x, uint y, int startZ, int endZ)
{
	for (int z = startZ; z < endZ; z++) {
		uint3 at = uint3(x, y, (uint)z);
		insideMask[at] = insideMask[at] + 1.0h;
	}
}
