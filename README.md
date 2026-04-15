# kandinsky.cpp

Native C/C++ inference runtime for Kandinsky 5 Pro text-to-video using `ggml` / `ggml-cuda` and GGUF weights.

This repository currently targets the official Kandinsky 5 Pro `5s` text-to-video checkpoint `kandinsky5pro_t2v_sft_5s.safetensors`.

Preconverted GGUF bundles:

- `https://huggingface.co/AkaneTendo25/K5-GGUF`

## Build

Windows + CUDA:

```powershell
git clone https://github.com/AkaneTendo25/kandinsky.cpp
cd kandinsky.cpp
git clone https://github.com/ggml-org/ggml G:\deps\ggml
set KD_GGML_DIR=G:\deps\ggml
cmd /c build_cuda.bat
```

Requires `CMake`, `Ninja`, `Visual Studio 2022 Build Tools`, and a CUDA toolkit visible to CMake.

If `ggml` is already present at `.\ggml`, `KD_GGML_DIR` is not needed.

```text
build_cuda\bin\kandinsky-cli.exe
```

## Run Inference

Example `5s` video run at `24 fps`:

```powershell
build_cuda\bin\kandinsky-cli.exe `
  -m <path-to-K5-GGUF>\q4 `
  -p "three energetic cats sprinting and weaving through fresh snow in a narrow alley, dynamic handheld camera following them, snow spraying everywhere, cinematic lighting, detailed fur, coherent motion" `
  -n "blurry, deformed, duplicate animals, extra limbs, low quality, washed out" `
  -o outputs\cats.mp4 `
  -W 512 -H 512 `
  --frames 121 `
  --fps 24 `
  -s 24 `
  --cfg-scale 5.0 `
  --scheduler-scale 5.0 `
  --seed 123 `
  -t 8 `
  --type q4_0 `
  --text-cpu
```

Notes:

- `121` frames at `24 fps` is about `5.04s`
- On the tested RTX 3090 machine, one `512x512 / 121-frame / 24-step / q4_0` clip takes about `30` minutes
- `q8_0` is supported, but it is significantly slower at this shape
- MP4 output expects `ffmpeg` in `PATH`

## Demo Videos

All clips below were generated with this runtime from the official-weight GGUF conversion.

| Video | Prompt | Parameters |
| --- | --- | --- |
| [Snowy alley cats MP4](samples/cats_5s_24fps.mp4) | `three energetic cats sprinting and weaving through fresh snow in a narrow alley, dynamic handheld camera following them, snow spraying everywhere, cinematic lighting, detailed fur, coherent motion`<br><br>Negative: `blurry, deformed, duplicate animals, extra limbs, low quality, washed out` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=123`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Motocross forest trail MP4](samples/motocross2_5s_24fps.mp4) | `two motocross riders racing through a muddy forest trail, dynamic tracking camera, dirt and water spraying, fast action, cinematic motion, detailed environment`<br><br>Negative: `blurry, deformed riders, duplicate bikes, extra wheels, low quality, washed out` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=321`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Birds over river forest MP4](samples/birds_5s_24fps.mp4) | `a flock of birds flying over a winding river through dense forest, sweeping aerial camera, morning mist, sunlight shafts through the trees, coherent motion`<br><br>Negative: `blurry, deformed birds, duplicate wings, extra limbs, low quality, washed out` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=888`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Blonde beach close-up MP4](samples/blonde_beach_5s_24fps.mp4) | `cinematic close-up of an adult blonde woman on a windy beach at golden hour, ocean behind her, detailed skin, natural expression, hair moving in the sea breeze, subtle handheld camera, coherent motion`<br><br>Negative: `blurry, deformed face, extra limbs, bad anatomy, low quality, washed out` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=4242`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Airplane flying MP4](samples/airplane_flying_5s_24fps.mp4) | `a silver passenger airplane flying above sunlit clouds at golden hour, banking smoothly through the sky, cinematic aerial tracking shot, realistic atmosphere, detailed wings and engines, coherent motion`<br><br>Negative: `blurry, deformed airplane, duplicate aircraft, broken wings, text, watermark, low quality, washed out` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=2026`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Small child girl smiling MP4](samples/small_child_girl_smiling_5s_24fps.mp4) | `a small child girl smiling warmly at the camera in a sunny park, gentle breeze moving her hair, natural soft daylight, realistic skin, expressive eyes, subtle head movement, cinematic close shot, coherent motion`<br><br>Negative: `blurry, deformed face, extra fingers, duplicate person, text, watermark, low quality, uncanny eyes, washed out` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=2027`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Space station explosion MP4](samples/space_station_explosion_5s_24fps.mp4) | `a massive space station exploding in deep space, expanding debris field, bright shockwave, glowing fragments, cinematic camera drift, stars and nebulae in the background, coherent motion`<br><br>Negative: `blurry, deformed structure, duplicate debris, low quality, washed out` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=9090`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Rally canyon chase MP4](samples/rally_5s_24fps.mp4) | `a red rally car drifting at high speed through a desert canyon road, dust and gravel spraying, cinematic chase camera, intense sunlight, coherent motion`<br><br>Negative: `blurry, deformed car, duplicate wheels, low quality, washed out` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=5151`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Fighter jets MP4](samples/jets_5s_24fps.mp4) | `fighter jets dogfighting above storm clouds, fast camera, cinematic action`<br><br>Negative: `blurry, low quality` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=8181`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Motocross mud jump MP4](samples/motocross_5s_24fps.mp4) | `motocross riders jumping through mud, fast camera, cinematic action`<br><br>Negative: `blurry, low quality` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=9393`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Man close-up MP4](samples/man_closeup_5s_24fps.mp4) | `close-up of an adult man, natural expression, subtle camera motion`<br><br>Negative: `blurry, low quality` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=5050`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Modern city aerial MP4](samples/modern_city_5s_24fps.mp4) | `aerial shot over a modern city, moving traffic, sweeping camera, cinematic motion`<br><br>Negative: `blurry, low quality` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=6060`, `type=q4_0`, `threads=8`, `text_cpu=on` |
| [Rural landscape aerial MP4](samples/rural_landscape_5s_24fps.mp4) | `rural landscape, rolling hills, winding road, cinematic motion`<br><br>Negative: `blurry, low quality` | `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=7070`, `type=q4_0`, `threads=8`, `text_cpu=on` |
