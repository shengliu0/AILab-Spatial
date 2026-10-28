// Golden values for the Kotlin ProcTex port: the noise primitives of NewtonLab.ProcTex copied
// verbatim with UnityEngine's Vector3/Mathf replaced by System.Math equivalents (Mathf.* are thin
// float casts of System.Math, FloorToInt = (int)Math.Floor). Plus System.Random sequences.
using System;
using System.Globalization;
static class N {
    static float Hash(int x, int y, int z, int seed) {
        unchecked {
            int h = seed;
            h = h * 73856093 ^ x * 19349663 ^ y * 83492791 ^ z * unchecked((int)2971215073u);
            h ^= h >> 13; h *= 1274126177; h ^= h >> 16;
            return (h & 0x7fffffff) / (float)0x7fffffff;
        }
    }
    static float Lerp(float a, float b, float t) { t = t < 0 ? 0 : t > 1 ? 1 : t; return a + (b - a) * t; }
    static void CellPoint(int x, int y, int z, int seed, out float cx, out float cy, out float cz) {
        cx = x + Hash(x, y, z, seed); cy = y + Hash(x, y, z, seed + 1); cz = z + Hash(x, y, z, seed + 2);
    }
    static void Worley(float px, float py, float pz, int seed, out float f1, out float f2) {
        int xi = (int)Math.Floor(px), yi = (int)Math.Floor(py), zi = (int)Math.Floor(pz);
        f1 = f2 = 1e9f;
        for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            float cx, cy, cz; CellPoint(xi + dx, yi + dy, zi + dz, seed, out cx, out cy, out cz);
            float ex = cx - px, ey = cy - py, ez = cz - pz;
            float d = ex * ex + ey * ey + ez * ez;
            if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) f2 = d;
        }
        f1 = (float)Math.Sqrt(f1); f2 = (float)Math.Sqrt(f2);
    }
    static float ValueNoise(float px, float py, float pz, int seed) {
        int xi = (int)Math.Floor(px), yi = (int)Math.Floor(py), zi = (int)Math.Floor(pz);
        float fx = px - xi, fy = py - yi, fz = pz - zi;
        fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
        float c000 = Hash(xi, yi, zi, seed), c100 = Hash(xi + 1, yi, zi, seed);
        float c010 = Hash(xi, yi + 1, zi, seed), c110 = Hash(xi + 1, yi + 1, zi, seed);
        float c001 = Hash(xi, yi, zi + 1, seed), c101 = Hash(xi + 1, yi, zi + 1, seed);
        float c011 = Hash(xi, yi + 1, zi + 1, seed), c111 = Hash(xi + 1, yi + 1, zi + 1, seed);
        float x00 = Lerp(c000, c100, fx), x10 = Lerp(c010, c110, fx);
        float x01 = Lerp(c001, c101, fx), x11 = Lerp(c011, c111, fx);
        return Lerp(Lerp(x00, x10, fy), Lerp(x01, x11, fy), fz);
    }
    static float Fbm(float px, float py, float pz, int octaves, int seed) {
        float sum = 0f, amp = 0.5f, norm = 0f;
        for (int i = 0; i < octaves; i++) {
            sum += ValueNoise(px, py, pz, seed + i * 101) * amp;
            norm += amp; amp *= 0.5f; px *= 2.02f; py *= 2.02f; pz *= 2.02f;
        }
        return norm > 0 ? sum / norm : 0f;
    }
    static string R(float f) { return f.ToString("R", CultureInfo.InvariantCulture); }
    static void Main() {
        Console.WriteLine("kind,a,b,c,d,e,value1,value2");
        var pts = new float[][] { new float[]{0.1f,0.2f,0.3f}, new float[]{-3.7f,12.25f,0.5f}, new float[]{88.8f,-41.3f,7.77f}, new float[]{0f,0f,0f}, new float[]{1234.5f,-0.001f,999.9f} };
        int[] seeds = { 12345, 20260731, 9000 };
        foreach (var s in seeds) foreach (var p in pts) {
            Console.WriteLine("hash,{0},{1},{2},{3},0,{4},0", (int)Math.Floor(p[0]), (int)Math.Floor(p[1]), (int)Math.Floor(p[2]), s, R(Hash((int)Math.Floor(p[0]), (int)Math.Floor(p[1]), (int)Math.Floor(p[2]), s)));
            float f1, f2; Worley(p[0], p[1], p[2], s, out f1, out f2);
            Console.WriteLine("worley,{0},{1},{2},{3},0,{4},{5}", R(p[0]), R(p[1]), R(p[2]), s, R(f1), R(f2));
            Console.WriteLine("fbm,{0},{1},{2},{3},4,{4},0", R(p[0]), R(p[1]), R(p[2]), s, R(Fbm(p[0], p[1], p[2], 4, s)));
        }
        foreach (var s in new int[]{20260731, 12345}) {
            var rng = new Random(s);
            for (int i = 0; i < 20; i++) Console.WriteLine("rand,{0},{1},0,0,0,{2},0", s, i, rng.NextDouble().ToString("R", CultureInfo.InvariantCulture));
        }
    }
}
