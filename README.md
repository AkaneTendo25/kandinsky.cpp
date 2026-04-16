# kandinsky.cpp

Native C/C++ inference runtime for [Kandinsky 5.0 Video Pro](https://github.com/kandinskylab/kandinsky-5) using `ggml` / `ggml-cuda` and GGUF weights.

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

### Snowy Alley Cats

https://github.com/user-attachments/assets/42a60ee4-b0c3-4e44-988d-888d64c5ff57

Prompt: `three energetic cats sprinting and weaving through fresh snow in a narrow alley, dynamic handheld camera following them, snow spraying everywhere, cinematic lighting, detailed fur, coherent motion`

Negative: `blurry, deformed, duplicate animals, extra limbs, low quality, washed out`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=123`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Motocross Forest Trail

https://github.com/user-attachments/assets/f6bbd5b8-6dd5-45f2-9e89-b679b5ce9efe

Prompt: `two motocross riders racing through a muddy forest trail, dynamic tracking camera, dirt and water spraying, fast action, cinematic motion, detailed environment`

Negative: `blurry, deformed riders, duplicate bikes, extra wheels, low quality, washed out`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=321`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Birds Over River Forest

https://github.com/user-attachments/assets/4312c0fb-87f6-493f-a265-4239c723bca1

Prompt: `a flock of birds flying over a winding river through dense forest, sweeping aerial camera, morning mist, sunlight shafts through the trees, coherent motion`

Negative: `blurry, deformed birds, duplicate wings, extra limbs, low quality, washed out`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=888`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Blonde Beach Close-Up

https://github.com/user-attachments/assets/bed3ee4a-5682-455c-9ba4-89ff044a51ac

Prompt: `cinematic close-up of an adult blonde woman on a windy beach at golden hour, ocean behind her, detailed skin, natural expression, hair moving in the sea breeze, subtle handheld camera, coherent motion`

Negative: `blurry, deformed face, extra limbs, bad anatomy, low quality, washed out`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=4242`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Airplane Flying

https://github.com/user-attachments/assets/67f85c1f-c2b9-45c8-b5a8-faf82335b80f

Prompt: `a silver passenger airplane flying above sunlit clouds at golden hour, banking smoothly through the sky, cinematic aerial tracking shot, realistic atmosphere, detailed wings and engines, coherent motion`

Negative: `blurry, deformed airplane, duplicate aircraft, broken wings, text, watermark, low quality, washed out`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=2026`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Small Child Girl Smiling

https://github.com/user-attachments/assets/97a5b283-1acb-4a49-a99e-2288fd3c5aae

Prompt: `a small child girl smiling warmly at the camera in a sunny park, gentle breeze moving her hair, natural soft daylight, realistic skin, expressive eyes, subtle head movement, cinematic close shot, coherent motion`

Negative: `blurry, deformed face, extra fingers, duplicate person, text, watermark, low quality, uncanny eyes, washed out`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=2027`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Space Station Explosion

https://github.com/user-attachments/assets/25f3c254-8790-4462-8e56-89cce9b4b3da

Prompt: `a massive space station exploding in deep space, expanding debris field, bright shockwave, glowing fragments, cinematic camera drift, stars and nebulae in the background, coherent motion`

Negative: `blurry, deformed structure, duplicate debris, low quality, washed out`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=9090`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Rally Canyon Chase

https://github.com/user-attachments/assets/956f00e2-ab00-4360-beeb-a28bd50e4190

Prompt: `a red rally car drifting at high speed through a desert canyon road, dust and gravel spraying, cinematic chase camera, intense sunlight, coherent motion`

Negative: `blurry, deformed car, duplicate wheels, low quality, washed out`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=5151`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Fighter Jets

https://github.com/user-attachments/assets/e5581dd3-3d11-4e72-9072-02d1b552fc9a

Prompt: `fighter jets dogfighting above storm clouds, fast camera, cinematic action`

Negative: `blurry, low quality`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=8181`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Motocross Mud Jump

https://github.com/user-attachments/assets/2078bd3d-eab5-48d4-b7b1-1353039f2f16

Prompt: `motocross riders jumping through mud, fast camera, cinematic action`

Negative: `blurry, low quality`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=9393`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Man Close-Up

https://github.com/user-attachments/assets/67ae7153-c2e4-4e65-8b0a-7ea4339e618d

Prompt: `close-up of an adult man, natural expression, subtle camera motion`

Negative: `blurry, low quality`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=5050`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Modern City Aerial

https://github.com/user-attachments/assets/6bb632e6-5d7c-4471-a22e-73b510e7c10c

Prompt: `aerial shot over a modern city, moving traffic, sweeping camera, cinematic motion`

Negative: `blurry, low quality`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=6060`, `type=q4_0`, `threads=8`, `text_cpu=on`

### Rural Landscape Aerial

https://github.com/user-attachments/assets/3d8d3942-7450-4395-a83d-7afd9eed959a

Prompt: `rural landscape, rolling hills, winding road, cinematic motion`

Negative: `blurry, low quality`

Parameters: `512x512`, `121` frames, `24 fps`, `24` steps, `cfg=5.0`, `scheduler=5.0`, `seed=7070`, `type=q4_0`, `threads=8`, `text_cpu=on`
