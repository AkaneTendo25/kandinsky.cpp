@echo off
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "BUILD_DIR=%ROOT%\build_cuda"
set "LOG_FILE=%BUILD_DIR%\build_log.txt"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VCVARS%" (
    echo Missing Visual Studio environment script: "%VCVARS%"
    exit /b 1
)

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo Failed to initialize the Visual Studio build environment
    exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if errorlevel 1 (
    echo Failed to create build directory "%BUILD_DIR%"
    exit /b 1
)

set "CMAKE_ARGS=-DKD_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF"
if defined KD_GGML_DIR (
    set "CMAKE_ARGS=%CMAKE_ARGS% -DKD_GGML_DIR=%KD_GGML_DIR%"
)

echo === CMAKE CONFIGURE === > "%LOG_FILE%"
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja %CMAKE_ARGS% >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo CONFIGURE FAILED >> "%LOG_FILE%"
    type "%LOG_FILE%"
    exit /b 1
)

echo === CMAKE BUILD === >> "%LOG_FILE%"
cmake --build "%BUILD_DIR%" --config RelWithDebInfo >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo BUILD FAILED >> "%LOG_FILE%"
    type "%LOG_FILE%"
    exit /b 1
)

echo BUILD COMPLETE >> "%LOG_FILE%"
type "%LOG_FILE%"
