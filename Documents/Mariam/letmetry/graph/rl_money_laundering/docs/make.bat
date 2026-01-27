@ECHO OFF

set SPHINXBUILD=python -m sphinx
set SOURCEDIR=.
set BUILDDIR=_build

if "%1"=="" (
    call %SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%
    goto end
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR%

:end
