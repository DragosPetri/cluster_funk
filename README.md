# SETUP

## Conan

[Install Guide](https://docs.conan.io/2/installation.html)

### clang profile for conan 

```
[settings]
os=Linux
arch=x86_64
build_type=Debug
compiler=clang
compiler.version=19
compiler.libcxx=libc++
compiler.cppstd=23

[platform_tool_requires]
cmake/[>=3.29]

[conf]
tools.build:compiler_executables={"c": "clang-19", "cpp": "clang++-19"}
tools.cmake.cmaketoolchain:generator=Ninja
tools.cmake.cmaketoolchain:extra_variables={'CMAKE_COLOR_DIAGNOSTICS': 'ON', 'CMAKE_EXPORT_COMPILE_COMMANDS': 'ON'}
```

## Clang

```
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 19
```

For clangd

`
ln -s path/to/repo/build/Debug/compile_commands.json path/to/repo/
`

If it's not enough maybe:

`
sudo apt install clang-19 libc++-19-dev libc++abi-19-dev clang-tools-19 clang-19-doc libclang-common-19-dev libclang-19-dev libclang1-19 clang-format-19 python3-clang-19 clangd-19 clang-tidy-19
`

## Handy zsh aliases

```
alias ci-clang="conan install . -pr=clang"
alias cb-clang="conan build . -pr=clang -c tools.build:skip_test=true"
alias ct-clang="conan build . -pr=clang"
alias cib-clang="ci-clang && ct-clang"
```
