# Detect Operating System
ifeq ($(OS),Windows_NT)
    # Windows settings
    RM := del /F /Q
    EXE_EXT := .exe
    ARCH := $(PROCESSOR_ARCHITECTURE)
    LNK_FLAGS := -luser32 -lgdi32 -lshell32
else
    # Unix/Linux settings
    RM := rm -f
    EXE_EXT :=
    ARCH := $(shell uname -m)
    LNK_FLAGS :=
endif

# Compiler and flags
CXX      := clang++
CXXFLAGS := -O3 -march=native -fno-finite-math-only -funroll-loops -flto -std=c++20 -DNDEBUG

IS_ARM := $(filter ARM arm64 aarch64 arm%,$(ARCH))

ifeq ($(IS_ARM),)
  CXXFLAGS += -static -fuse-ld=lld
endif


# Default target executable name and evaluation file path
EXE      ?= Neuro$(EXE_EXT)

# Source and object files
SRCS     := $(wildcard ./src/*.cpp)
OBJS     := $(SRCS:.cpp=.o)

# Default target
all: $(EXE)

# Link the executable
$(EXE): $(SRCS)
	$(CXX) $(CXXFLAGS) $(LNK_FLAGS) $(SRCS) ./external/fmt/format.cc -I ./external/ -o $@

# Debug build
.PHONY: debug
debug: clean
debug: CXXFLAGS = -O3 -ggdb -march=native -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -fuse-ld=lld -fno-finite-math-only -flto -std=c++20 -Wall -Wextra
debug: all

# Debug build
.PHONY: profile
profile: clean
profile: CXXFLAGS = -O2 -ggdb -march=native -fno-finite-math-only -funroll-loops -flto -std=c++20 -fno-omit-frame-pointer -fuse-ld=lld -DNDEBUG
profile: all

# Force rebuild
.PHONY: force
force: clean all

# Clean up
.PHONY: clean
clean:
	$(RM) $(EXE)
	$(RM) Neuro.exp
	$(RM) Neuro.lib
	$(RM) Neuro.pdb