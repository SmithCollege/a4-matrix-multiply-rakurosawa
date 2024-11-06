## Raw Data Collection

| Test Data | cpuMatMul (ns) | gpuMatMul (ns) |  tiledMM (ns) | cublasMM (ns) |
|-----------|----------------|----------------|---------------|---------------|
| Array of size 128 | 8333921.432495 | 354051.589966 | 359058.380127 | 971078.872681 |
| Array of size 256 | 86591958.999634 | 660896.301270 | 586032.867432 | 1057863.235474 |
| Array of size 512 | 793749094.009399 | 1897096.633911 | 1175880.432129 | 2272129.058838 |
| Array of size 1024 | 6435665130.615234 | 11639833.450317 | 4843235.015869 | 3184080.123901 |

![](https://github.com/SmithCollege/a4-matrix-multiply-rakurosawa/blob/de4697483e1e0f2845e73df3b5de9c9a7f07472e/Time%20vs%20Size%201.png)
![](https://github.com/SmithCollege/a4-matrix-multiply-rakurosawa/blob/de4697483e1e0f2845e73df3b5de9c9a7f07472e/Time%20vs%20Size%202.png)
