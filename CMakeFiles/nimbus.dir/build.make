# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/irfan/projects/nimbus

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/irfan/projects/nimbus

# Include any dependencies generated for this target.
include CMakeFiles/nimbus.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/nimbus.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/nimbus.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nimbus.dir/flags.make

CMakeFiles/nimbus.dir/codegen:
.PHONY : CMakeFiles/nimbus.dir/codegen

CMakeFiles/nimbus.dir/main.cpp.o: CMakeFiles/nimbus.dir/flags.make
CMakeFiles/nimbus.dir/main.cpp.o: main.cpp
CMakeFiles/nimbus.dir/main.cpp.o: CMakeFiles/nimbus.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/irfan/projects/nimbus/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/nimbus.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/nimbus.dir/main.cpp.o -MF CMakeFiles/nimbus.dir/main.cpp.o.d -o CMakeFiles/nimbus.dir/main.cpp.o -c /Users/irfan/projects/nimbus/main.cpp

CMakeFiles/nimbus.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/nimbus.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/irfan/projects/nimbus/main.cpp > CMakeFiles/nimbus.dir/main.cpp.i

CMakeFiles/nimbus.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/nimbus.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/irfan/projects/nimbus/main.cpp -o CMakeFiles/nimbus.dir/main.cpp.s

# Object files for target nimbus
nimbus_OBJECTS = \
"CMakeFiles/nimbus.dir/main.cpp.o"

# External object files for target nimbus
nimbus_EXTERNAL_OBJECTS =

nimbus: CMakeFiles/nimbus.dir/main.cpp.o
nimbus: CMakeFiles/nimbus.dir/build.make
nimbus: libmodels.a
nimbus: /opt/homebrew/lib/libtensorflow.dylib
nimbus: CMakeFiles/nimbus.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/irfan/projects/nimbus/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable nimbus"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nimbus.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nimbus.dir/build: nimbus
.PHONY : CMakeFiles/nimbus.dir/build

CMakeFiles/nimbus.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nimbus.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nimbus.dir/clean

CMakeFiles/nimbus.dir/depend:
	cd /Users/irfan/projects/nimbus && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/irfan/projects/nimbus /Users/irfan/projects/nimbus /Users/irfan/projects/nimbus /Users/irfan/projects/nimbus /Users/irfan/projects/nimbus/CMakeFiles/nimbus.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/nimbus.dir/depend

