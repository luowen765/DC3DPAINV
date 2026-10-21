MFEM_DIR ?= /usr/local/mfem
EIGEN_INCLUDE ?= -I/usr/include/eigen3

ifneq ($(filter clean,$(MAKECMDGOALS)),)
else
ifeq ($(wildcard $(MFEM_DIR)/config/config.mk),)
$(error MFEM_DIR does not point to an MFEM build directory: $(MFEM_DIR))
endif
include $(MFEM_DIR)/config/config.mk
CXX = $(MFEM_CXX)
CXXFLAGS = $(MFEM_FLAGS)
endif

SRC = src
DC3D_INCLUDE = -I ../../$(SRC)/
INCLUDES = $(wildcard ../../$(SRC)/*.h)
SRCS = $(wildcard ../../$(SRC)/*.cpp)
OBJS = $(patsubst %.cpp, %.o, $(SRCS))

%.o : %.cpp $(INCLUDES)
	@echo "DC3DInv is compiling "$<"..."
	@$(CXX) $(CXXFLAGS) $(DC3D_INCLUDE) $(EIGEN_INCLUDE) $(MFEM_INCLUDE) -c $< -o $@

DC3DInv: $(OBJS)
	@$(CXX) $(CXXFLAGS) -o DC3DInv $(OBJS) $(MFEM_LIBS)

clean:
	@rm -rf $(OBJS) DC3DInv
