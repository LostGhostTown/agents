@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cls

echo ======================================================
echo          LangGraph Agent 项目 一键环境配置
echo          自动创建同名虚拟环境 | 适配GitHub部署
echo ======================================================
echo.

:: ==============================================
:: 1. 自动获取 当前文件夹名称 作为虚拟环境名
:: ==============================================
for %%i in ("%cd%") do set "VENV_NAME=%%~ni"
echo [1/5] 检测项目名称：!VENV_NAME!
echo [1/5] 即将创建虚拟环境：!VENV_NAME!
echo.

:: ==============================================
:: 2. 创建虚拟环境（如果不存在）
:: ==============================================
if not exist "!VENV_NAME!\Scripts\activate.bat" (
    echo [2/5] 正在创建虚拟环境...
    python -m venv !VENV_NAME!
    if errorlevel 1 (
        echo.
        echo ❌ 虚拟环境创建失败！请检查Python是否安装
        pause >nul
        exit /b 1
    )
    echo ✅ 虚拟环境创建成功
) else (
    echo [2/5] 虚拟环境已存在，跳过创建
)
echo.

:: ==============================================
:: 3. 激活虚拟环境
:: ==============================================
echo [3/5] 激活虚拟环境...
call "!VENV_NAME!\Scripts\activate.bat"
echo ✅ 虚拟环境已激活
echo.

:: ==============================================
:: 4. 升级pip + 安装所有依赖
:: ==============================================
echo [4/5] 升级pip...
python -m pip install --upgrade pip >nul
echo ✅ pip升级完成

echo [4/5] 安装项目依赖（requirements.txt）...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ 依赖安装失败
    pause >nul
    exit /b 1
)

echo [4/5] 安装项目（可编辑模式，解决导入报错）...
pip install -e . >nul
echo ✅ 依赖&项目配置完成
echo.

:: ==============================================
:: 5. 完成提示
:: ==============================================
echo ======================================================
echo                 🎉 环境配置全部完成
echo ======================================================
echo.
echo 启动命令：langgraph dev
echo 虚拟环境已激活，可直接运行！
echo.
echo ======================================================
pause >nul