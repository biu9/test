??	
??
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
i
LinSpace

start"T	
stop"T
num"Tidx
output"T"
Ttype:
2"
Tidxtype0:
2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
?
	MirrorPad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	"&
modestring:
REFLECT	SYMMETRIC
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResizeBilinear
images"T
size
resized_images"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
H
ShardedFilename
basename	
shard

num_shards
filename
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.14.02unknown??	
l
PlaceholderPlaceholder*&
_output_shapes
:*
dtype0*
shape:
?
gen_flows/ConstConst*A
value8B6"            @           @   @   @*&
_output_shapes
:*
dtype0
q
gen_flows/Tile/multiplesConst*%
valueB"            *
_output_shapes
:*
dtype0
?
gen_flows/TileTilegen_flows/Constgen_flows/Tile/multiples*
T0*&
_output_shapes
:*

Tmultiples0
p
gen_flows/layer_0/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
o
%gen_flows/layer_0/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
q
'gen_flows/layer_0/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
q
'gen_flows/layer_0/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
gen_flows/layer_0/strided_sliceStridedSlicegen_flows/layer_0/Shape%gen_flows/layer_0/strided_slice/stack'gen_flows/layer_0/strided_slice/stack_1'gen_flows/layer_0/strided_slice/stack_2*
new_axis_mask *
_output_shapes
: *

begin_mask *
Index0*
ellipsis_mask *
end_mask *
shrink_axis_mask*
T0
Y
gen_flows/layer_0/mul/xConst*
value	B :*
_output_shapes
: *
dtype0
w
gen_flows/layer_0/mulMulgen_flows/layer_0/mul/xgen_flows/layer_0/strided_slice*
_output_shapes
: *
T0
r
gen_flows/layer_0/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
q
'gen_flows/layer_0/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0
s
)gen_flows/layer_0/strided_slice_1/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
s
)gen_flows/layer_0/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
!gen_flows/layer_0/strided_slice_1StridedSlicegen_flows/layer_0/Shape_1'gen_flows/layer_0/strided_slice_1/stack)gen_flows/layer_0/strided_slice_1/stack_1)gen_flows/layer_0/strided_slice_1/stack_2*
ellipsis_mask *
_output_shapes
: *
T0*
Index0*

begin_mask *
shrink_axis_mask*
new_axis_mask *
end_mask 
[
gen_flows/layer_0/mul_1/xConst*
dtype0*
value	B :*
_output_shapes
: 
}
gen_flows/layer_0/mul_1Mulgen_flows/layer_0/mul_1/x!gen_flows/layer_0/strided_slice_1*
T0*
_output_shapes
: 
?
.gen_flows/layer_0/upsample2d_layer/resize/sizePackgen_flows/layer_0/mulgen_flows/layer_0/mul_1*
_output_shapes
:*

axis *
N*
T0
?
8gen_flows/layer_0/upsample2d_layer/resize/ResizeBilinearResizeBilinearPlaceholder.gen_flows/layer_0/upsample2d_layer/resize/size*
align_corners(*
T0*
half_pixel_centers( *&
_output_shapes
:
?
$gen_flows/layer_0/pad_layer/paddingsConst*9
value0B."                               *
_output_shapes

:*
dtype0
?
gen_flows/layer_0/pad_layer	MirrorPad8gen_flows/layer_0/upsample2d_layer/resize/ResizeBilinear$gen_flows/layer_0/pad_layer/paddings*&
_output_shapes
:*
mode	REFLECT*
	Tpaddings0*
T0
?
?gen_flows/layer_0/conv2d/kernel/Initializer/random_normal/shapeConst*%
valueB"         ?   *
dtype0*
_output_shapes
:*2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel
?
>gen_flows/layer_0/conv2d/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel*
valueB
 *    *
dtype0
?
@gen_flows/layer_0/conv2d/kernel/Initializer/random_normal/stddevConst*
valueB
 *
ף<*2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel*
_output_shapes
: *
dtype0
?
Ngen_flows/layer_0/conv2d/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal?gen_flows/layer_0/conv2d/kernel/Initializer/random_normal/shape*

seed *
T0*2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel*
dtype0*'
_output_shapes
:?*
seed2 
?
=gen_flows/layer_0/conv2d/kernel/Initializer/random_normal/mulMulNgen_flows/layer_0/conv2d/kernel/Initializer/random_normal/RandomStandardNormal@gen_flows/layer_0/conv2d/kernel/Initializer/random_normal/stddev*2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel*'
_output_shapes
:?*
T0
?
9gen_flows/layer_0/conv2d/kernel/Initializer/random_normalAdd=gen_flows/layer_0/conv2d/kernel/Initializer/random_normal/mul>gen_flows/layer_0/conv2d/kernel/Initializer/random_normal/mean*2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel*'
_output_shapes
:?*
T0
?
gen_flows/layer_0/conv2d/kernel
VariableV2*
shape:?*'
_output_shapes
:?*
	container *
dtype0*
shared_name *2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel
?
&gen_flows/layer_0/conv2d/kernel/AssignAssigngen_flows/layer_0/conv2d/kernel9gen_flows/layer_0/conv2d/kernel/Initializer/random_normal*
T0*'
_output_shapes
:?*2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel*
use_locking(*
validate_shape(
?
$gen_flows/layer_0/conv2d/kernel/readIdentitygen_flows/layer_0/conv2d/kernel*'
_output_shapes
:?*2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel*
T0
?
/gen_flows/layer_0/conv2d/bias/Initializer/ConstConst*
valueB?*    *
dtype0*
_output_shapes	
:?*0
_class&
$"loc:@gen_flows/layer_0/conv2d/bias
?
gen_flows/layer_0/conv2d/bias
VariableV2*
shape:?*
_output_shapes	
:?*
	container *
shared_name *
dtype0*0
_class&
$"loc:@gen_flows/layer_0/conv2d/bias
?
$gen_flows/layer_0/conv2d/bias/AssignAssigngen_flows/layer_0/conv2d/bias/gen_flows/layer_0/conv2d/bias/Initializer/Const*0
_class&
$"loc:@gen_flows/layer_0/conv2d/bias*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0
?
"gen_flows/layer_0/conv2d/bias/readIdentitygen_flows/layer_0/conv2d/bias*
_output_shapes	
:?*
T0*0
_class&
$"loc:@gen_flows/layer_0/conv2d/bias
w
&gen_flows/layer_0/conv2d/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
gen_flows/layer_0/conv2d/Conv2DConv2Dgen_flows/layer_0/pad_layer$gen_flows/layer_0/conv2d/kernel/read*'
_output_shapes
:?*
explicit_paddings
 *
T0*
	dilations
*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
 gen_flows/layer_0/conv2d/BiasAddBiasAddgen_flows/layer_0/conv2d/Conv2D"gen_flows/layer_0/conv2d/bias/read*'
_output_shapes
:?*
data_formatNHWC*
T0
?
"gen_flows/layer_0/conv2d/LeakyRelu	LeakyRelu gen_flows/layer_0/conv2d/BiasAdd*'
_output_shapes
:?*
alpha%??L>*
T0
?
&gen_flows/layer_0/pad_layer_1/paddingsConst*9
value0B."                               *
_output_shapes

:*
dtype0
?
gen_flows/layer_0/pad_layer_1	MirrorPadgen_flows/Tile&gen_flows/layer_0/pad_layer_1/paddings*&
_output_shapes
:*
	Tpaddings0*
mode	REFLECT*
T0
n
#gen_flows/layer_0/concat_layer/axisConst*
valueB :
?????????*
_output_shapes
: *
dtype0
?
gen_flows/layer_0/concat_layerConcatV2"gen_flows/layer_0/conv2d/LeakyRelugen_flows/layer_0/pad_layer_1#gen_flows/layer_0/concat_layer/axis*

Tidx0*
N*
T0*'
_output_shapes
:?
p
gen_flows/layer_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         ?   
o
%gen_flows/layer_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
q
'gen_flows/layer_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
q
'gen_flows/layer_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
gen_flows/layer_1/strided_sliceStridedSlicegen_flows/layer_1/Shape%gen_flows/layer_1/strided_slice/stack'gen_flows/layer_1/strided_slice/stack_1'gen_flows/layer_1/strided_slice/stack_2*
end_mask *
ellipsis_mask *
shrink_axis_mask*
new_axis_mask *
Index0*
T0*
_output_shapes
: *

begin_mask 
Y
gen_flows/layer_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
w
gen_flows/layer_1/mulMulgen_flows/layer_1/mul/xgen_flows/layer_1/strided_slice*
T0*
_output_shapes
: 
r
gen_flows/layer_1/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"         ?   
q
'gen_flows/layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
s
)gen_flows/layer_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)gen_flows/layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
!gen_flows/layer_1/strided_slice_1StridedSlicegen_flows/layer_1/Shape_1'gen_flows/layer_1/strided_slice_1/stack)gen_flows/layer_1/strided_slice_1/stack_1)gen_flows/layer_1/strided_slice_1/stack_2*
new_axis_mask *
_output_shapes
: *
Index0*
T0*

begin_mask *
end_mask *
ellipsis_mask *
shrink_axis_mask
[
gen_flows/layer_1/mul_1/xConst*
value	B :*
dtype0*
_output_shapes
: 
}
gen_flows/layer_1/mul_1Mulgen_flows/layer_1/mul_1/x!gen_flows/layer_1/strided_slice_1*
_output_shapes
: *
T0
?
.gen_flows/layer_1/upsample2d_layer/resize/sizePackgen_flows/layer_1/mulgen_flows/layer_1/mul_1*

axis *
_output_shapes
:*
T0*
N
?
8gen_flows/layer_1/upsample2d_layer/resize/ResizeBilinearResizeBilineargen_flows/layer_0/concat_layer.gen_flows/layer_1/upsample2d_layer/resize/size*'
_output_shapes
:?*
T0*
align_corners(*
half_pixel_centers( 
?
$gen_flows/layer_1/pad_layer/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
?
gen_flows/layer_1/pad_layer	MirrorPad8gen_flows/layer_1/upsample2d_layer/resize/ResizeBilinear$gen_flows/layer_1/pad_layer/paddings*
	Tpaddings0*'
_output_shapes
:		?*
T0*
mode	REFLECT
?
?gen_flows/layer_1/conv2d/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   *2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel
?
>gen_flows/layer_1/conv2d/kernel/Initializer/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: *2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel
?
@gen_flows/layer_1/conv2d/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<*2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel
?
Ngen_flows/layer_1/conv2d/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal?gen_flows/layer_1/conv2d/kernel/Initializer/random_normal/shape*2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel*

seed *
dtype0*
T0*(
_output_shapes
:??*
seed2 
?
=gen_flows/layer_1/conv2d/kernel/Initializer/random_normal/mulMulNgen_flows/layer_1/conv2d/kernel/Initializer/random_normal/RandomStandardNormal@gen_flows/layer_1/conv2d/kernel/Initializer/random_normal/stddev*(
_output_shapes
:??*
T0*2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel
?
9gen_flows/layer_1/conv2d/kernel/Initializer/random_normalAdd=gen_flows/layer_1/conv2d/kernel/Initializer/random_normal/mul>gen_flows/layer_1/conv2d/kernel/Initializer/random_normal/mean*2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel*
T0*(
_output_shapes
:??
?
gen_flows/layer_1/conv2d/kernel
VariableV2*(
_output_shapes
:??*
dtype0*
shared_name *2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel*
	container *
shape:??
?
&gen_flows/layer_1/conv2d/kernel/AssignAssigngen_flows/layer_1/conv2d/kernel9gen_flows/layer_1/conv2d/kernel/Initializer/random_normal*
validate_shape(*2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel*
use_locking(*
T0*(
_output_shapes
:??
?
$gen_flows/layer_1/conv2d/kernel/readIdentitygen_flows/layer_1/conv2d/kernel*(
_output_shapes
:??*
T0*2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel
?
/gen_flows/layer_1/conv2d/bias/Initializer/ConstConst*
valueB?*    *
dtype0*
_output_shapes	
:?*0
_class&
$"loc:@gen_flows/layer_1/conv2d/bias
?
gen_flows/layer_1/conv2d/bias
VariableV2*
_output_shapes	
:?*
shared_name *
dtype0*0
_class&
$"loc:@gen_flows/layer_1/conv2d/bias*
	container *
shape:?
?
$gen_flows/layer_1/conv2d/bias/AssignAssigngen_flows/layer_1/conv2d/bias/gen_flows/layer_1/conv2d/bias/Initializer/Const*
validate_shape(*0
_class&
$"loc:@gen_flows/layer_1/conv2d/bias*
use_locking(*
T0*
_output_shapes	
:?
?
"gen_flows/layer_1/conv2d/bias/readIdentitygen_flows/layer_1/conv2d/bias*
_output_shapes	
:?*
T0*0
_class&
$"loc:@gen_flows/layer_1/conv2d/bias
w
&gen_flows/layer_1/conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
gen_flows/layer_1/conv2d/Conv2DConv2Dgen_flows/layer_1/pad_layer$gen_flows/layer_1/conv2d/kernel/read*
paddingVALID*
explicit_paddings
 *
strides
*
	dilations
*
data_formatNHWC*
use_cudnn_on_gpu(*'
_output_shapes
:?*
T0
?
 gen_flows/layer_1/conv2d/BiasAddBiasAddgen_flows/layer_1/conv2d/Conv2D"gen_flows/layer_1/conv2d/bias/read*'
_output_shapes
:?*
T0*
data_formatNHWC
?
"gen_flows/layer_1/conv2d/LeakyRelu	LeakyRelu gen_flows/layer_1/conv2d/BiasAdd*'
_output_shapes
:?*
alpha%??L>*
T0
p
gen_flows/layer_2/ShapeConst*%
valueB"         ?   *
dtype0*
_output_shapes
:
o
%gen_flows/layer_2/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
q
'gen_flows/layer_2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
q
'gen_flows/layer_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
gen_flows/layer_2/strided_sliceStridedSlicegen_flows/layer_2/Shape%gen_flows/layer_2/strided_slice/stack'gen_flows/layer_2/strided_slice/stack_1'gen_flows/layer_2/strided_slice/stack_2*
new_axis_mask *
_output_shapes
: *
Index0*
T0*

begin_mask *
end_mask *
ellipsis_mask *
shrink_axis_mask
Y
gen_flows/layer_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
w
gen_flows/layer_2/mulMulgen_flows/layer_2/mul/xgen_flows/layer_2/strided_slice*
_output_shapes
: *
T0
r
gen_flows/layer_2/Shape_1Const*%
valueB"         ?   *
dtype0*
_output_shapes
:
q
'gen_flows/layer_2/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
s
)gen_flows/layer_2/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
s
)gen_flows/layer_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
!gen_flows/layer_2/strided_slice_1StridedSlicegen_flows/layer_2/Shape_1'gen_flows/layer_2/strided_slice_1/stack)gen_flows/layer_2/strided_slice_1/stack_1)gen_flows/layer_2/strided_slice_1/stack_2*
end_mask *
new_axis_mask *

begin_mask *
shrink_axis_mask*
ellipsis_mask *
_output_shapes
: *
T0*
Index0
[
gen_flows/layer_2/mul_1/xConst*
dtype0*
_output_shapes
: *
value	B :
}
gen_flows/layer_2/mul_1Mulgen_flows/layer_2/mul_1/x!gen_flows/layer_2/strided_slice_1*
_output_shapes
: *
T0
?
.gen_flows/layer_2/upsample2d_layer/resize/sizePackgen_flows/layer_2/mulgen_flows/layer_2/mul_1*

axis *
_output_shapes
:*
N*
T0
?
8gen_flows/layer_2/upsample2d_layer/resize/ResizeBilinearResizeBilinear"gen_flows/layer_1/conv2d/LeakyRelu.gen_flows/layer_2/upsample2d_layer/resize/size*'
_output_shapes
:?*
T0*
half_pixel_centers( *
align_corners(
?
$gen_flows/layer_2/pad_layer/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
?
gen_flows/layer_2/pad_layer	MirrorPad8gen_flows/layer_2/upsample2d_layer/resize/ResizeBilinear$gen_flows/layer_2/pad_layer/paddings*
	Tpaddings0*
T0*'
_output_shapes
:?*
mode	REFLECT
?
?gen_flows/layer_2/conv2d/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel*
dtype0*%
valueB"      ?   ?   
?
>gen_flows/layer_2/conv2d/kernel/Initializer/random_normal/meanConst*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel*
dtype0*
_output_shapes
: *
valueB
 *    
?
@gen_flows/layer_2/conv2d/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel*
dtype0*
valueB
 *
ף<
?
Ngen_flows/layer_2/conv2d/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal?gen_flows/layer_2/conv2d/kernel/Initializer/random_normal/shape*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel*

seed *
dtype0*(
_output_shapes
:??*
T0*
seed2 
?
=gen_flows/layer_2/conv2d/kernel/Initializer/random_normal/mulMulNgen_flows/layer_2/conv2d/kernel/Initializer/random_normal/RandomStandardNormal@gen_flows/layer_2/conv2d/kernel/Initializer/random_normal/stddev*(
_output_shapes
:??*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel*
T0
?
9gen_flows/layer_2/conv2d/kernel/Initializer/random_normalAdd=gen_flows/layer_2/conv2d/kernel/Initializer/random_normal/mul>gen_flows/layer_2/conv2d/kernel/Initializer/random_normal/mean*(
_output_shapes
:??*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel*
T0
?
gen_flows/layer_2/conv2d/kernel
VariableV2*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel*
shape:??*(
_output_shapes
:??*
	container *
shared_name *
dtype0
?
&gen_flows/layer_2/conv2d/kernel/AssignAssigngen_flows/layer_2/conv2d/kernel9gen_flows/layer_2/conv2d/kernel/Initializer/random_normal*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:??*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel
?
$gen_flows/layer_2/conv2d/kernel/readIdentitygen_flows/layer_2/conv2d/kernel*(
_output_shapes
:??*
T0*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel
?
/gen_flows/layer_2/conv2d/bias/Initializer/ConstConst*
valueB?*    *0
_class&
$"loc:@gen_flows/layer_2/conv2d/bias*
dtype0*
_output_shapes	
:?
?
gen_flows/layer_2/conv2d/bias
VariableV2*
shared_name *0
_class&
$"loc:@gen_flows/layer_2/conv2d/bias*
shape:?*
dtype0*
	container *
_output_shapes	
:?
?
$gen_flows/layer_2/conv2d/bias/AssignAssigngen_flows/layer_2/conv2d/bias/gen_flows/layer_2/conv2d/bias/Initializer/Const*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0*0
_class&
$"loc:@gen_flows/layer_2/conv2d/bias
?
"gen_flows/layer_2/conv2d/bias/readIdentitygen_flows/layer_2/conv2d/bias*
_output_shapes	
:?*
T0*0
_class&
$"loc:@gen_flows/layer_2/conv2d/bias
w
&gen_flows/layer_2/conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
gen_flows/layer_2/conv2d/Conv2DConv2Dgen_flows/layer_2/pad_layer$gen_flows/layer_2/conv2d/kernel/read*
data_formatNHWC*
explicit_paddings
 *'
_output_shapes
:?*
strides
*
T0*
	dilations
*
use_cudnn_on_gpu(*
paddingVALID
?
 gen_flows/layer_2/conv2d/BiasAddBiasAddgen_flows/layer_2/conv2d/Conv2D"gen_flows/layer_2/conv2d/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:?
?
"gen_flows/layer_2/conv2d/LeakyRelu	LeakyRelu gen_flows/layer_2/conv2d/BiasAdd*
T0*'
_output_shapes
:?*
alpha%??L>
p
gen_flows/layer_3/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"         ?   
o
%gen_flows/layer_3/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
q
'gen_flows/layer_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
q
'gen_flows/layer_3/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
gen_flows/layer_3/strided_sliceStridedSlicegen_flows/layer_3/Shape%gen_flows/layer_3/strided_slice/stack'gen_flows/layer_3/strided_slice/stack_1'gen_flows/layer_3/strided_slice/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0*
end_mask *

begin_mask *
ellipsis_mask *
new_axis_mask 
Y
gen_flows/layer_3/mul/xConst*
dtype0*
_output_shapes
: *
value	B :
w
gen_flows/layer_3/mulMulgen_flows/layer_3/mul/xgen_flows/layer_3/strided_slice*
T0*
_output_shapes
: 
r
gen_flows/layer_3/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"         ?   
q
'gen_flows/layer_3/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
s
)gen_flows/layer_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
s
)gen_flows/layer_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
!gen_flows/layer_3/strided_slice_1StridedSlicegen_flows/layer_3/Shape_1'gen_flows/layer_3/strided_slice_1/stack)gen_flows/layer_3/strided_slice_1/stack_1)gen_flows/layer_3/strided_slice_1/stack_2*
Index0*
_output_shapes
: *
T0*

begin_mask *
end_mask *
new_axis_mask *
ellipsis_mask *
shrink_axis_mask
[
gen_flows/layer_3/mul_1/xConst*
value	B :*
dtype0*
_output_shapes
: 
}
gen_flows/layer_3/mul_1Mulgen_flows/layer_3/mul_1/x!gen_flows/layer_3/strided_slice_1*
_output_shapes
: *
T0
?
.gen_flows/layer_3/upsample2d_layer/resize/sizePackgen_flows/layer_3/mulgen_flows/layer_3/mul_1*
_output_shapes
:*
N*
T0*

axis 
?
8gen_flows/layer_3/upsample2d_layer/resize/ResizeBilinearResizeBilinear"gen_flows/layer_2/conv2d/LeakyRelu.gen_flows/layer_3/upsample2d_layer/resize/size*
T0*'
_output_shapes
:?*
align_corners(*
half_pixel_centers( 
?
$gen_flows/layer_3/pad_layer/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
gen_flows/layer_3/pad_layer	MirrorPad8gen_flows/layer_3/upsample2d_layer/resize/ResizeBilinear$gen_flows/layer_3/pad_layer/paddings*
mode	REFLECT*
T0*
	Tpaddings0*'
_output_shapes
:!!?
?
?gen_flows/layer_3/conv2d/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel*%
valueB"      ?   @   
?
>gen_flows/layer_3/conv2d/kernel/Initializer/random_normal/meanConst*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel*
dtype0*
valueB
 *    *
_output_shapes
: 
?
@gen_flows/layer_3/conv2d/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel
?
Ngen_flows/layer_3/conv2d/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal?gen_flows/layer_3/conv2d/kernel/Initializer/random_normal/shape*
dtype0*
T0*'
_output_shapes
:?@*
seed2 *

seed *2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel
?
=gen_flows/layer_3/conv2d/kernel/Initializer/random_normal/mulMulNgen_flows/layer_3/conv2d/kernel/Initializer/random_normal/RandomStandardNormal@gen_flows/layer_3/conv2d/kernel/Initializer/random_normal/stddev*
T0*'
_output_shapes
:?@*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel
?
9gen_flows/layer_3/conv2d/kernel/Initializer/random_normalAdd=gen_flows/layer_3/conv2d/kernel/Initializer/random_normal/mul>gen_flows/layer_3/conv2d/kernel/Initializer/random_normal/mean*
T0*'
_output_shapes
:?@*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel
?
gen_flows/layer_3/conv2d/kernel
VariableV2*'
_output_shapes
:?@*
	container *
dtype0*
shared_name *
shape:?@*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel
?
&gen_flows/layer_3/conv2d/kernel/AssignAssigngen_flows/layer_3/conv2d/kernel9gen_flows/layer_3/conv2d/kernel/Initializer/random_normal*
validate_shape(*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel*
T0*
use_locking(*'
_output_shapes
:?@
?
$gen_flows/layer_3/conv2d/kernel/readIdentitygen_flows/layer_3/conv2d/kernel*
T0*'
_output_shapes
:?@*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel
?
/gen_flows/layer_3/conv2d/bias/Initializer/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@*0
_class&
$"loc:@gen_flows/layer_3/conv2d/bias
?
gen_flows/layer_3/conv2d/bias
VariableV2*0
_class&
$"loc:@gen_flows/layer_3/conv2d/bias*
shape:@*
_output_shapes
:@*
shared_name *
dtype0*
	container 
?
$gen_flows/layer_3/conv2d/bias/AssignAssigngen_flows/layer_3/conv2d/bias/gen_flows/layer_3/conv2d/bias/Initializer/Const*
use_locking(*
T0*
_output_shapes
:@*0
_class&
$"loc:@gen_flows/layer_3/conv2d/bias*
validate_shape(
?
"gen_flows/layer_3/conv2d/bias/readIdentitygen_flows/layer_3/conv2d/bias*
_output_shapes
:@*
T0*0
_class&
$"loc:@gen_flows/layer_3/conv2d/bias
w
&gen_flows/layer_3/conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
gen_flows/layer_3/conv2d/Conv2DConv2Dgen_flows/layer_3/pad_layer$gen_flows/layer_3/conv2d/kernel/read*
use_cudnn_on_gpu(*
strides
*&
_output_shapes
:@*
explicit_paddings
 *
T0*
	dilations
*
paddingVALID*
data_formatNHWC
?
 gen_flows/layer_3/conv2d/BiasAddBiasAddgen_flows/layer_3/conv2d/Conv2D"gen_flows/layer_3/conv2d/bias/read*
data_formatNHWC*
T0*&
_output_shapes
:@
?
"gen_flows/layer_3/conv2d/LeakyRelu	LeakyRelu gen_flows/layer_3/conv2d/BiasAdd*
alpha%??L>*&
_output_shapes
:@*
T0
p
gen_flows/layer_4/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
o
%gen_flows/layer_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
q
'gen_flows/layer_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
q
'gen_flows/layer_4/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
gen_flows/layer_4/strided_sliceStridedSlicegen_flows/layer_4/Shape%gen_flows/layer_4/strided_slice/stack'gen_flows/layer_4/strided_slice/stack_1'gen_flows/layer_4/strided_slice/stack_2*

begin_mask *
end_mask *
ellipsis_mask *
new_axis_mask *
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
Y
gen_flows/layer_4/mul/xConst*
dtype0*
value	B :*
_output_shapes
: 
w
gen_flows/layer_4/mulMulgen_flows/layer_4/mul/xgen_flows/layer_4/strided_slice*
T0*
_output_shapes
: 
r
gen_flows/layer_4/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         @   
q
'gen_flows/layer_4/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
s
)gen_flows/layer_4/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
s
)gen_flows/layer_4/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
!gen_flows/layer_4/strided_slice_1StridedSlicegen_flows/layer_4/Shape_1'gen_flows/layer_4/strided_slice_1/stack)gen_flows/layer_4/strided_slice_1/stack_1)gen_flows/layer_4/strided_slice_1/stack_2*
Index0*
_output_shapes
: *
ellipsis_mask *
end_mask *
T0*
new_axis_mask *

begin_mask *
shrink_axis_mask
[
gen_flows/layer_4/mul_1/xConst*
value	B :*
dtype0*
_output_shapes
: 
}
gen_flows/layer_4/mul_1Mulgen_flows/layer_4/mul_1/x!gen_flows/layer_4/strided_slice_1*
T0*
_output_shapes
: 
?
.gen_flows/layer_4/upsample2d_layer/resize/sizePackgen_flows/layer_4/mulgen_flows/layer_4/mul_1*
N*

axis *
T0*
_output_shapes
:
?
8gen_flows/layer_4/upsample2d_layer/resize/ResizeBilinearResizeBilinear"gen_flows/layer_3/conv2d/LeakyRelu.gen_flows/layer_4/upsample2d_layer/resize/size*
half_pixel_centers( *
T0*&
_output_shapes
:>>@*
align_corners(
?
$gen_flows/layer_4/pad_layer/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
gen_flows/layer_4/pad_layer	MirrorPad8gen_flows/layer_4/upsample2d_layer/resize/ResizeBilinear$gen_flows/layer_4/pad_layer/paddings*
mode	REFLECT*
T0*
	Tpaddings0*&
_output_shapes
:AA@
?
?gen_flows/layer_4/conv2d/kernel/Initializer/random_normal/shapeConst*2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel*
dtype0*%
valueB"      @   @   *
_output_shapes
:
?
>gen_flows/layer_4/conv2d/kernel/Initializer/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: *2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel
?
@gen_flows/layer_4/conv2d/kernel/Initializer/random_normal/stddevConst*
valueB
 *
ף<*
dtype0*
_output_shapes
: *2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel
?
Ngen_flows/layer_4/conv2d/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal?gen_flows/layer_4/conv2d/kernel/Initializer/random_normal/shape*
dtype0*&
_output_shapes
:@@*
T0*
seed2 *

seed *2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel
?
=gen_flows/layer_4/conv2d/kernel/Initializer/random_normal/mulMulNgen_flows/layer_4/conv2d/kernel/Initializer/random_normal/RandomStandardNormal@gen_flows/layer_4/conv2d/kernel/Initializer/random_normal/stddev*
T0*&
_output_shapes
:@@*2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel
?
9gen_flows/layer_4/conv2d/kernel/Initializer/random_normalAdd=gen_flows/layer_4/conv2d/kernel/Initializer/random_normal/mul>gen_flows/layer_4/conv2d/kernel/Initializer/random_normal/mean*2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel*
T0*&
_output_shapes
:@@
?
gen_flows/layer_4/conv2d/kernel
VariableV2*
shared_name *
shape:@@*2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel*
	container *
dtype0*&
_output_shapes
:@@
?
&gen_flows/layer_4/conv2d/kernel/AssignAssigngen_flows/layer_4/conv2d/kernel9gen_flows/layer_4/conv2d/kernel/Initializer/random_normal*
T0*
validate_shape(*&
_output_shapes
:@@*2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel*
use_locking(
?
$gen_flows/layer_4/conv2d/kernel/readIdentitygen_flows/layer_4/conv2d/kernel*
T0*2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel*&
_output_shapes
:@@
?
/gen_flows/layer_4/conv2d/bias/Initializer/ConstConst*
valueB@*    *0
_class&
$"loc:@gen_flows/layer_4/conv2d/bias*
dtype0*
_output_shapes
:@
?
gen_flows/layer_4/conv2d/bias
VariableV2*0
_class&
$"loc:@gen_flows/layer_4/conv2d/bias*
dtype0*
	container *
_output_shapes
:@*
shared_name *
shape:@
?
$gen_flows/layer_4/conv2d/bias/AssignAssigngen_flows/layer_4/conv2d/bias/gen_flows/layer_4/conv2d/bias/Initializer/Const*0
_class&
$"loc:@gen_flows/layer_4/conv2d/bias*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0
?
"gen_flows/layer_4/conv2d/bias/readIdentitygen_flows/layer_4/conv2d/bias*
T0*0
_class&
$"loc:@gen_flows/layer_4/conv2d/bias*
_output_shapes
:@
w
&gen_flows/layer_4/conv2d/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
gen_flows/layer_4/conv2d/Conv2DConv2Dgen_flows/layer_4/pad_layer$gen_flows/layer_4/conv2d/kernel/read*
data_formatNHWC*
strides
*
explicit_paddings
 *
T0*
	dilations
*
paddingVALID*&
_output_shapes
:??@*
use_cudnn_on_gpu(
?
 gen_flows/layer_4/conv2d/BiasAddBiasAddgen_flows/layer_4/conv2d/Conv2D"gen_flows/layer_4/conv2d/bias/read*
T0*&
_output_shapes
:??@*
data_formatNHWC
?
"gen_flows/layer_4/conv2d/LeakyRelu	LeakyRelu gen_flows/layer_4/conv2d/BiasAdd*
T0*&
_output_shapes
:??@*
alpha%??L>
p
gen_flows/layer_5/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?   ?   @   
o
%gen_flows/layer_5/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
q
'gen_flows/layer_5/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
q
'gen_flows/layer_5/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
gen_flows/layer_5/strided_sliceStridedSlicegen_flows/layer_5/Shape%gen_flows/layer_5/strided_slice/stack'gen_flows/layer_5/strided_slice/stack_1'gen_flows/layer_5/strided_slice/stack_2*
Index0*
shrink_axis_mask*
ellipsis_mask *
end_mask *
new_axis_mask *
T0*
_output_shapes
: *

begin_mask 
Y
gen_flows/layer_5/mul/xConst*
_output_shapes
: *
value	B :*
dtype0
w
gen_flows/layer_5/mulMulgen_flows/layer_5/mul/xgen_flows/layer_5/strided_slice*
T0*
_output_shapes
: 
r
gen_flows/layer_5/Shape_1Const*%
valueB"   ?   ?   @   *
dtype0*
_output_shapes
:
q
'gen_flows/layer_5/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
s
)gen_flows/layer_5/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)gen_flows/layer_5/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
!gen_flows/layer_5/strided_slice_1StridedSlicegen_flows/layer_5/Shape_1'gen_flows/layer_5/strided_slice_1/stack)gen_flows/layer_5/strided_slice_1/stack_1)gen_flows/layer_5/strided_slice_1/stack_2*
Index0*
shrink_axis_mask*
ellipsis_mask *
end_mask *
new_axis_mask *
T0*
_output_shapes
: *

begin_mask 
[
gen_flows/layer_5/mul_1/xConst*
_output_shapes
: *
value	B :*
dtype0
}
gen_flows/layer_5/mul_1Mulgen_flows/layer_5/mul_1/x!gen_flows/layer_5/strided_slice_1*
T0*
_output_shapes
: 
?
.gen_flows/layer_5/upsample2d_layer/resize/sizePackgen_flows/layer_5/mulgen_flows/layer_5/mul_1*
_output_shapes
:*
T0*
N*

axis 
?
8gen_flows/layer_5/upsample2d_layer/resize/ResizeBilinearResizeBilinear"gen_flows/layer_4/conv2d/LeakyRelu.gen_flows/layer_5/upsample2d_layer/resize/size*
align_corners(*
half_pixel_centers( *
T0*&
_output_shapes
:~~@
?
$gen_flows/layer_5/pad_layer/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
gen_flows/layer_5/pad_layer	MirrorPad8gen_flows/layer_5/upsample2d_layer/resize/ResizeBilinear$gen_flows/layer_5/pad_layer/paddings*
	Tpaddings0*
mode	REFLECT*
T0*(
_output_shapes
:??@
?
?gen_flows/layer_5/conv2d/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*%
valueB"      @   @   *
dtype0
?
>gen_flows/layer_5/conv2d/kernel/Initializer/random_normal/meanConst*
valueB
 *    *2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*
dtype0*
_output_shapes
: 
?
@gen_flows/layer_5/conv2d/kernel/Initializer/random_normal/stddevConst*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*
dtype0*
_output_shapes
: *
valueB
 *
ף<
?
Ngen_flows/layer_5/conv2d/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal?gen_flows/layer_5/conv2d/kernel/Initializer/random_normal/shape*

seed *
dtype0*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*
T0*&
_output_shapes
:@@*
seed2 
?
=gen_flows/layer_5/conv2d/kernel/Initializer/random_normal/mulMulNgen_flows/layer_5/conv2d/kernel/Initializer/random_normal/RandomStandardNormal@gen_flows/layer_5/conv2d/kernel/Initializer/random_normal/stddev*
T0*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*&
_output_shapes
:@@
?
9gen_flows/layer_5/conv2d/kernel/Initializer/random_normalAdd=gen_flows/layer_5/conv2d/kernel/Initializer/random_normal/mul>gen_flows/layer_5/conv2d/kernel/Initializer/random_normal/mean*
T0*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*&
_output_shapes
:@@
?
gen_flows/layer_5/conv2d/kernel
VariableV2*
dtype0*
shared_name *
shape:@@*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*&
_output_shapes
:@@*
	container 
?
&gen_flows/layer_5/conv2d/kernel/AssignAssigngen_flows/layer_5/conv2d/kernel9gen_flows/layer_5/conv2d/kernel/Initializer/random_normal*
T0*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:@@
?
$gen_flows/layer_5/conv2d/kernel/readIdentitygen_flows/layer_5/conv2d/kernel*
T0*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*&
_output_shapes
:@@
?
/gen_flows/layer_5/conv2d/bias/Initializer/ConstConst*
dtype0*0
_class&
$"loc:@gen_flows/layer_5/conv2d/bias*
valueB@*    *
_output_shapes
:@
?
gen_flows/layer_5/conv2d/bias
VariableV2*0
_class&
$"loc:@gen_flows/layer_5/conv2d/bias*
dtype0*
	container *
_output_shapes
:@*
shared_name *
shape:@
?
$gen_flows/layer_5/conv2d/bias/AssignAssigngen_flows/layer_5/conv2d/bias/gen_flows/layer_5/conv2d/bias/Initializer/Const*
T0*0
_class&
$"loc:@gen_flows/layer_5/conv2d/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
?
"gen_flows/layer_5/conv2d/bias/readIdentitygen_flows/layer_5/conv2d/bias*0
_class&
$"loc:@gen_flows/layer_5/conv2d/bias*
T0*
_output_shapes
:@
w
&gen_flows/layer_5/conv2d/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
gen_flows/layer_5/conv2d/Conv2DConv2Dgen_flows/layer_5/pad_layer$gen_flows/layer_5/conv2d/kernel/read*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
T0*
explicit_paddings
 *
paddingVALID*
	dilations
*&
_output_shapes
:~~@
?
 gen_flows/layer_5/conv2d/BiasAddBiasAddgen_flows/layer_5/conv2d/Conv2D"gen_flows/layer_5/conv2d/bias/read*
T0*&
_output_shapes
:~~@*
data_formatNHWC
?
"gen_flows/layer_5/conv2d/LeakyRelu	LeakyRelu gen_flows/layer_5/conv2d/BiasAdd*
T0*&
_output_shapes
:~~@*
alpha%??L>
p
gen_flows/layer_6/ShapeConst*%
valueB"   ~   ~   @   *
_output_shapes
:*
dtype0
o
%gen_flows/layer_6/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
q
'gen_flows/layer_6/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
q
'gen_flows/layer_6/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
gen_flows/layer_6/strided_sliceStridedSlicegen_flows/layer_6/Shape%gen_flows/layer_6/strided_slice/stack'gen_flows/layer_6/strided_slice/stack_1'gen_flows/layer_6/strided_slice/stack_2*
Index0*
_output_shapes
: *
ellipsis_mask *
end_mask *
T0*
new_axis_mask *

begin_mask *
shrink_axis_mask
Y
gen_flows/layer_6/mul/xConst*
value	B :*
_output_shapes
: *
dtype0
w
gen_flows/layer_6/mulMulgen_flows/layer_6/mul/xgen_flows/layer_6/strided_slice*
_output_shapes
: *
T0
r
gen_flows/layer_6/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"   ~   ~   @   
q
'gen_flows/layer_6/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
s
)gen_flows/layer_6/strided_slice_1/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
s
)gen_flows/layer_6/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
!gen_flows/layer_6/strided_slice_1StridedSlicegen_flows/layer_6/Shape_1'gen_flows/layer_6/strided_slice_1/stack)gen_flows/layer_6/strided_slice_1/stack_1)gen_flows/layer_6/strided_slice_1/stack_2*
T0*

begin_mask *
new_axis_mask *
_output_shapes
: *
Index0*
ellipsis_mask *
shrink_axis_mask*
end_mask 
[
gen_flows/layer_6/mul_1/xConst*
value	B :*
_output_shapes
: *
dtype0
}
gen_flows/layer_6/mul_1Mulgen_flows/layer_6/mul_1/x!gen_flows/layer_6/strided_slice_1*
T0*
_output_shapes
: 
?
.gen_flows/layer_6/upsample2d_layer/resize/sizePackgen_flows/layer_6/mulgen_flows/layer_6/mul_1*
N*
_output_shapes
:*
T0*

axis 
?
8gen_flows/layer_6/upsample2d_layer/resize/ResizeBilinearResizeBilinear"gen_flows/layer_5/conv2d/LeakyRelu.gen_flows/layer_6/upsample2d_layer/resize/size*
T0*
align_corners(*
half_pixel_centers( *(
_output_shapes
:??@
?
$gen_flows/layer_6/pad_layer/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
?
gen_flows/layer_6/pad_layer	MirrorPad8gen_flows/layer_6/upsample2d_layer/resize/ResizeBilinear$gen_flows/layer_6/pad_layer/paddings*(
_output_shapes
:??@*
	Tpaddings0*
mode	REFLECT*
T0
?
?gen_flows/layer_6/conv2d/kernel/Initializer/random_normal/shapeConst*%
valueB"      @       *
_output_shapes
:*
dtype0*2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel
?
>gen_flows/layer_6/conv2d/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel*
valueB
 *    *
dtype0
?
@gen_flows/layer_6/conv2d/kernel/Initializer/random_normal/stddevConst*
valueB
 *
ף<*
_output_shapes
: *
dtype0*2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel
?
Ngen_flows/layer_6/conv2d/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal?gen_flows/layer_6/conv2d/kernel/Initializer/random_normal/shape*

seed *2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel*
T0*&
_output_shapes
:@ *
seed2 *
dtype0
?
=gen_flows/layer_6/conv2d/kernel/Initializer/random_normal/mulMulNgen_flows/layer_6/conv2d/kernel/Initializer/random_normal/RandomStandardNormal@gen_flows/layer_6/conv2d/kernel/Initializer/random_normal/stddev*&
_output_shapes
:@ *
T0*2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel
?
9gen_flows/layer_6/conv2d/kernel/Initializer/random_normalAdd=gen_flows/layer_6/conv2d/kernel/Initializer/random_normal/mul>gen_flows/layer_6/conv2d/kernel/Initializer/random_normal/mean*
T0*&
_output_shapes
:@ *2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel
?
gen_flows/layer_6/conv2d/kernel
VariableV2*
	container *
shape:@ *2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel*&
_output_shapes
:@ *
dtype0*
shared_name 
?
&gen_flows/layer_6/conv2d/kernel/AssignAssigngen_flows/layer_6/conv2d/kernel9gen_flows/layer_6/conv2d/kernel/Initializer/random_normal*
use_locking(*&
_output_shapes
:@ *
validate_shape(*2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel*
T0
?
$gen_flows/layer_6/conv2d/kernel/readIdentitygen_flows/layer_6/conv2d/kernel*&
_output_shapes
:@ *
T0*2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel
?
/gen_flows/layer_6/conv2d/bias/Initializer/ConstConst*
valueB *    *
_output_shapes
: *
dtype0*0
_class&
$"loc:@gen_flows/layer_6/conv2d/bias
?
gen_flows/layer_6/conv2d/bias
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
	container *
shape: *0
_class&
$"loc:@gen_flows/layer_6/conv2d/bias
?
$gen_flows/layer_6/conv2d/bias/AssignAssigngen_flows/layer_6/conv2d/bias/gen_flows/layer_6/conv2d/bias/Initializer/Const*
use_locking(*
_output_shapes
: *
validate_shape(*0
_class&
$"loc:@gen_flows/layer_6/conv2d/bias*
T0
?
"gen_flows/layer_6/conv2d/bias/readIdentitygen_flows/layer_6/conv2d/bias*
_output_shapes
: *
T0*0
_class&
$"loc:@gen_flows/layer_6/conv2d/bias
w
&gen_flows/layer_6/conv2d/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
gen_flows/layer_6/conv2d/Conv2DConv2Dgen_flows/layer_6/pad_layer$gen_flows/layer_6/conv2d/kernel/read*
paddingVALID*(
_output_shapes
:?? *
	dilations
*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
T0*
explicit_paddings
 
?
 gen_flows/layer_6/conv2d/BiasAddBiasAddgen_flows/layer_6/conv2d/Conv2D"gen_flows/layer_6/conv2d/bias/read*(
_output_shapes
:?? *
data_formatNHWC*
T0
?
"gen_flows/layer_6/conv2d/LeakyRelu	LeakyRelu gen_flows/layer_6/conv2d/BiasAdd*
alpha%??L>*(
_output_shapes
:?? *
T0
p
gen_flows/layer_7/ShapeConst*%
valueB"   ?   ?       *
_output_shapes
:*
dtype0
o
%gen_flows/layer_7/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
q
'gen_flows/layer_7/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
q
'gen_flows/layer_7/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
gen_flows/layer_7/strided_sliceStridedSlicegen_flows/layer_7/Shape%gen_flows/layer_7/strided_slice/stack'gen_flows/layer_7/strided_slice/stack_1'gen_flows/layer_7/strided_slice/stack_2*

begin_mask *
new_axis_mask *
Index0*
_output_shapes
: *
ellipsis_mask *
end_mask *
shrink_axis_mask*
T0
Y
gen_flows/layer_7/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
w
gen_flows/layer_7/mulMulgen_flows/layer_7/mul/xgen_flows/layer_7/strided_slice*
T0*
_output_shapes
: 
r
gen_flows/layer_7/Shape_1Const*%
valueB"   ?   ?       *
dtype0*
_output_shapes
:
q
'gen_flows/layer_7/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
s
)gen_flows/layer_7/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)gen_flows/layer_7/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
!gen_flows/layer_7/strided_slice_1StridedSlicegen_flows/layer_7/Shape_1'gen_flows/layer_7/strided_slice_1/stack)gen_flows/layer_7/strided_slice_1/stack_1)gen_flows/layer_7/strided_slice_1/stack_2*

begin_mask *
new_axis_mask *
Index0*
_output_shapes
: *
ellipsis_mask *
end_mask *
shrink_axis_mask*
T0
[
gen_flows/layer_7/mul_1/xConst*
value	B :*
dtype0*
_output_shapes
: 
}
gen_flows/layer_7/mul_1Mulgen_flows/layer_7/mul_1/x!gen_flows/layer_7/strided_slice_1*
T0*
_output_shapes
: 
?
.gen_flows/layer_7/upsample2d_layer/resize/sizePackgen_flows/layer_7/mulgen_flows/layer_7/mul_1*
T0*
_output_shapes
:*
N*

axis 
?
8gen_flows/layer_7/upsample2d_layer/resize/ResizeBilinearResizeBilinear"gen_flows/layer_6/conv2d/LeakyRelu.gen_flows/layer_7/upsample2d_layer/resize/size*
T0*
align_corners(*
half_pixel_centers( *(
_output_shapes
:?? 
?
$gen_flows/layer_7/pad_layer/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
gen_flows/layer_7/pad_layer	MirrorPad8gen_flows/layer_7/upsample2d_layer/resize/ResizeBilinear$gen_flows/layer_7/pad_layer/paddings*
T0*(
_output_shapes
:?? *
mode	REFLECT*
	Tpaddings0
?
?gen_flows/layer_7/conv2d/kernel/Initializer/random_normal/shapeConst*2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel*
dtype0*
_output_shapes
:*%
valueB"              
?
>gen_flows/layer_7/conv2d/kernel/Initializer/random_normal/meanConst*
valueB
 *    *2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel*
_output_shapes
: *
dtype0
?
@gen_flows/layer_7/conv2d/kernel/Initializer/random_normal/stddevConst*2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel*
dtype0*
_output_shapes
: *
valueB
 *
ף<
?
Ngen_flows/layer_7/conv2d/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal?gen_flows/layer_7/conv2d/kernel/Initializer/random_normal/shape*
dtype0*&
_output_shapes
:  *
T0*
seed2 *

seed *2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel
?
=gen_flows/layer_7/conv2d/kernel/Initializer/random_normal/mulMulNgen_flows/layer_7/conv2d/kernel/Initializer/random_normal/RandomStandardNormal@gen_flows/layer_7/conv2d/kernel/Initializer/random_normal/stddev*
T0*2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel*&
_output_shapes
:  
?
9gen_flows/layer_7/conv2d/kernel/Initializer/random_normalAdd=gen_flows/layer_7/conv2d/kernel/Initializer/random_normal/mul>gen_flows/layer_7/conv2d/kernel/Initializer/random_normal/mean*&
_output_shapes
:  *2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel*
T0
?
gen_flows/layer_7/conv2d/kernel
VariableV2*
shape:  *
dtype0*&
_output_shapes
:  *
	container *
shared_name *2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel
?
&gen_flows/layer_7/conv2d/kernel/AssignAssigngen_flows/layer_7/conv2d/kernel9gen_flows/layer_7/conv2d/kernel/Initializer/random_normal*
validate_shape(*2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel*
use_locking(*
T0*&
_output_shapes
:  
?
$gen_flows/layer_7/conv2d/kernel/readIdentitygen_flows/layer_7/conv2d/kernel*
T0*2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel*&
_output_shapes
:  
?
/gen_flows/layer_7/conv2d/bias/Initializer/ConstConst*
valueB *    *0
_class&
$"loc:@gen_flows/layer_7/conv2d/bias*
dtype0*
_output_shapes
: 
?
gen_flows/layer_7/conv2d/bias
VariableV2*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name *0
_class&
$"loc:@gen_flows/layer_7/conv2d/bias
?
$gen_flows/layer_7/conv2d/bias/AssignAssigngen_flows/layer_7/conv2d/bias/gen_flows/layer_7/conv2d/bias/Initializer/Const*
validate_shape(*0
_class&
$"loc:@gen_flows/layer_7/conv2d/bias*
use_locking(*
_output_shapes
: *
T0
?
"gen_flows/layer_7/conv2d/bias/readIdentitygen_flows/layer_7/conv2d/bias*
T0*0
_class&
$"loc:@gen_flows/layer_7/conv2d/bias*
_output_shapes
: 
w
&gen_flows/layer_7/conv2d/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
gen_flows/layer_7/conv2d/Conv2DConv2Dgen_flows/layer_7/pad_layer$gen_flows/layer_7/conv2d/kernel/read*
explicit_paddings
 *
	dilations
*(
_output_shapes
:?? *
strides
*
paddingVALID*
T0*
data_formatNHWC*
use_cudnn_on_gpu(
?
 gen_flows/layer_7/conv2d/BiasAddBiasAddgen_flows/layer_7/conv2d/Conv2D"gen_flows/layer_7/conv2d/bias/read*(
_output_shapes
:?? *
T0*
data_formatNHWC
?
"gen_flows/layer_7/conv2d/LeakyRelu	LeakyRelu gen_flows/layer_7/conv2d/BiasAdd*
alpha%??L>*
T0*(
_output_shapes
:?? 
?
*gen_flows/outputs_flows/pad_layer/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
?
!gen_flows/outputs_flows/pad_layer	MirrorPad"gen_flows/layer_7/conv2d/LeakyRelu*gen_flows/outputs_flows/pad_layer/paddings*
	Tpaddings0*(
_output_shapes
:?? *
T0*
mode	REFLECT
?
Egen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal/shapeConst*%
valueB"             *8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel*
dtype0*
_output_shapes
:
?
Dgen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal/meanConst*
valueB
 *    *8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel*
_output_shapes
: *
dtype0
?
Fgen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal/stddevConst*
valueB
 *
ף<*8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel*
dtype0*
_output_shapes
: 
?
Tgen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalEgen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal/shape*
dtype0*
T0*&
_output_shapes
: *
seed2 *

seed *8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel
?
Cgen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal/mulMulTgen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal/RandomStandardNormalFgen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal/stddev*8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel*
T0*&
_output_shapes
: 
?
?gen_flows/outputs_flows/conv2d/kernel/Initializer/random_normalAddCgen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal/mulDgen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal/mean*8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel*&
_output_shapes
: *
T0
?
%gen_flows/outputs_flows/conv2d/kernel
VariableV2*
	container *8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel*
shape: *&
_output_shapes
: *
shared_name *
dtype0
?
,gen_flows/outputs_flows/conv2d/kernel/AssignAssign%gen_flows/outputs_flows/conv2d/kernel?gen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal*
validate_shape(*8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel*
use_locking(*&
_output_shapes
: *
T0
?
*gen_flows/outputs_flows/conv2d/kernel/readIdentity%gen_flows/outputs_flows/conv2d/kernel*8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel*&
_output_shapes
: *
T0
?
5gen_flows/outputs_flows/conv2d/bias/Initializer/ConstConst*
valueB*    *6
_class,
*(loc:@gen_flows/outputs_flows/conv2d/bias*
dtype0*
_output_shapes
:
?
#gen_flows/outputs_flows/conv2d/bias
VariableV2*
	container *6
_class,
*(loc:@gen_flows/outputs_flows/conv2d/bias*
shape:*
_output_shapes
:*
shared_name *
dtype0
?
*gen_flows/outputs_flows/conv2d/bias/AssignAssign#gen_flows/outputs_flows/conv2d/bias5gen_flows/outputs_flows/conv2d/bias/Initializer/Const*
T0*6
_class,
*(loc:@gen_flows/outputs_flows/conv2d/bias*
use_locking(*
_output_shapes
:*
validate_shape(
?
(gen_flows/outputs_flows/conv2d/bias/readIdentity#gen_flows/outputs_flows/conv2d/bias*
T0*6
_class,
*(loc:@gen_flows/outputs_flows/conv2d/bias*
_output_shapes
:
}
,gen_flows/outputs_flows/conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
%gen_flows/outputs_flows/conv2d/Conv2DConv2D!gen_flows/outputs_flows/pad_layer*gen_flows/outputs_flows/conv2d/kernel/read*
strides
*
	dilations
*
T0*(
_output_shapes
:??*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*
explicit_paddings
 
?
&gen_flows/outputs_flows/conv2d/BiasAddBiasAdd%gen_flows/outputs_flows/conv2d/Conv2D(gen_flows/outputs_flows/conv2d/bias/read*(
_output_shapes
:??*
data_formatNHWC*
T0
?
#gen_flows/outputs_flows/conv2d/TanhTanh&gen_flows/outputs_flows/conv2d/BiasAdd*
T0*(
_output_shapes
:??
n
ConstConst*(
_output_shapes
:??*
dtype0*'
valueB??*  ??
n
Placeholder_1Placeholder*&
_output_shapes
:*
shape:*
dtype0
r
Placeholder_2Placeholder*(
_output_shapes
:??*
dtype0*
shape:??
r
Placeholder_3Placeholder*(
_output_shapes
:??*
dtype0*
shape:??
d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
f
strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
f
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
?
strided_sliceStridedSlice#gen_flows/outputs_flows/conv2d/Tanhstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*

begin_mask*
shrink_axis_mask *
new_axis_mask *
end_mask*
ellipsis_mask *(
_output_shapes
:??*
Index0
W
subSubPlaceholderPlaceholder_1*&
_output_shapes
:*
T0
g
Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   ?  ?     
f
TileTilesubTile/multiples*

Tmultiples0*
T0*(
_output_shapes
:??
R
mulMulTilestrided_slice*
T0*(
_output_shapes
:??
[
truedivRealDivPlaceholder_2Const*
T0*(
_output_shapes
:??
o
bilinear_sampler/ShapeConst*%
valueB"   ?  ?     *
_output_shapes
:*
dtype0
n
$bilinear_sampler/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
p
&bilinear_sampler/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&bilinear_sampler/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
bilinear_sampler/strided_sliceStridedSlicebilinear_sampler/Shape$bilinear_sampler/strided_slice/stack&bilinear_sampler/strided_slice/stack_1&bilinear_sampler/strided_slice/stack_2*
T0*

begin_mask *
shrink_axis_mask*
new_axis_mask *
end_mask *
ellipsis_mask *
_output_shapes
: *
Index0
q
bilinear_sampler/Shape_1Const*%
valueB"   ?  ?     *
dtype0*
_output_shapes
:
p
&bilinear_sampler/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
r
(bilinear_sampler/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
r
(bilinear_sampler/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
 bilinear_sampler/strided_slice_1StridedSlicebilinear_sampler/Shape_1&bilinear_sampler/strided_slice_1/stack(bilinear_sampler/strided_slice_1/stack_1(bilinear_sampler/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
shrink_axis_mask*
end_mask *
ellipsis_mask *
new_axis_mask 
q
bilinear_sampler/Shape_2Const*
_output_shapes
:*
dtype0*%
valueB"   ?  ?     
p
&bilinear_sampler/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
r
(bilinear_sampler/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
r
(bilinear_sampler/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
 bilinear_sampler/strided_slice_2StridedSlicebilinear_sampler/Shape_2&bilinear_sampler/strided_slice_2/stack(bilinear_sampler/strided_slice_2/stack_1(bilinear_sampler/strided_slice_2/stack_2*
T0*
Index0*
_output_shapes
: *

begin_mask *
new_axis_mask *
end_mask *
shrink_axis_mask*
ellipsis_mask 
q
bilinear_sampler/Shape_3Const*
dtype0*%
valueB"   ?  ?     *
_output_shapes
:
p
&bilinear_sampler/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
r
(bilinear_sampler/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
r
(bilinear_sampler/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
 bilinear_sampler/strided_slice_3StridedSlicebilinear_sampler/Shape_3&bilinear_sampler/strided_slice_3/stack(bilinear_sampler/strided_slice_3/stack_1(bilinear_sampler/strided_slice_3/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: *

begin_mask *
end_mask *
ellipsis_mask *
new_axis_mask 

bilinear_sampler/CastCast bilinear_sampler/strided_slice_1*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
?
bilinear_sampler/Cast_1Cast bilinear_sampler/strided_slice_2*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
e
 bilinear_sampler/transform/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
bilinear_sampler/transform/subSubbilinear_sampler/Cast_1 bilinear_sampler/transform/sub/y*
T0*
_output_shapes
: 
n
)bilinear_sampler/transform/LinSpace/startConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
#bilinear_sampler/transform/LinSpaceLinSpace)bilinear_sampler/transform/LinSpace/startbilinear_sampler/transform/sub bilinear_sampler/strided_slice_2*
_output_shapes	
:?*
T0*

Tidx0
g
"bilinear_sampler/transform/sub_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
 bilinear_sampler/transform/sub_1Subbilinear_sampler/Cast"bilinear_sampler/transform/sub_1/y*
_output_shapes
: *
T0
p
+bilinear_sampler/transform/LinSpace_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
%bilinear_sampler/transform/LinSpace_1LinSpace+bilinear_sampler/transform/LinSpace_1/start bilinear_sampler/transform/sub_1 bilinear_sampler/strided_slice_1*
T0*
_output_shapes	
:?*

Tidx0
?
1bilinear_sampler/transform/meshgrid/Reshape/shapeConst*
_output_shapes
:*
valueB"????   *
dtype0
?
+bilinear_sampler/transform/meshgrid/ReshapeReshape#bilinear_sampler/transform/LinSpace1bilinear_sampler/transform/meshgrid/Reshape/shape*
_output_shapes
:	?*
Tshape0*
T0
?
3bilinear_sampler/transform/meshgrid/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????
?
-bilinear_sampler/transform/meshgrid/Reshape_1Reshape%bilinear_sampler/transform/LinSpace_13bilinear_sampler/transform/meshgrid/Reshape_1/shape*
Tshape0*
_output_shapes
:	?*
T0
k
(bilinear_sampler/transform/meshgrid/SizeConst*
_output_shapes
: *
value
B :?*
dtype0
m
*bilinear_sampler/transform/meshgrid/Size_1Const*
_output_shapes
: *
dtype0*
value
B :?
?
3bilinear_sampler/transform/meshgrid/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????
?
-bilinear_sampler/transform/meshgrid/Reshape_2Reshape+bilinear_sampler/transform/meshgrid/Reshape3bilinear_sampler/transform/meshgrid/Reshape_2/shape*
_output_shapes
:	?*
T0*
Tshape0
?
3bilinear_sampler/transform/meshgrid/Reshape_3/shapeConst*
_output_shapes
:*
valueB"????   *
dtype0
?
-bilinear_sampler/transform/meshgrid/Reshape_3Reshape-bilinear_sampler/transform/meshgrid/Reshape_13bilinear_sampler/transform/meshgrid/Reshape_3/shape*
T0*
_output_shapes
:	?*
Tshape0
?
,bilinear_sampler/transform/meshgrid/ones/mulMul*bilinear_sampler/transform/meshgrid/Size_1(bilinear_sampler/transform/meshgrid/Size*
T0*
_output_shapes
: 
r
/bilinear_sampler/transform/meshgrid/ones/Less/yConst*
dtype0*
value
B :?*
_output_shapes
: 
?
-bilinear_sampler/transform/meshgrid/ones/LessLess,bilinear_sampler/transform/meshgrid/ones/mul/bilinear_sampler/transform/meshgrid/ones/Less/y*
_output_shapes
: *
T0
?
/bilinear_sampler/transform/meshgrid/ones/packedPack*bilinear_sampler/transform/meshgrid/Size_1(bilinear_sampler/transform/meshgrid/Size*
T0*

axis *
N*
_output_shapes
:
s
.bilinear_sampler/transform/meshgrid/ones/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
(bilinear_sampler/transform/meshgrid/onesFill/bilinear_sampler/transform/meshgrid/ones/packed.bilinear_sampler/transform/meshgrid/ones/Const*

index_type0*
T0* 
_output_shapes
:
??
?
'bilinear_sampler/transform/meshgrid/mulMul-bilinear_sampler/transform/meshgrid/Reshape_2(bilinear_sampler/transform/meshgrid/ones*
T0* 
_output_shapes
:
??
?
)bilinear_sampler/transform/meshgrid/mul_1Mul-bilinear_sampler/transform/meshgrid/Reshape_3(bilinear_sampler/transform/meshgrid/ones* 
_output_shapes
:
??*
T0
y
(bilinear_sampler/transform/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"   ????
?
"bilinear_sampler/transform/ReshapeReshape'bilinear_sampler/transform/meshgrid/mul(bilinear_sampler/transform/Reshape/shape*
T0* 
_output_shapes
:
??*
Tshape0
{
*bilinear_sampler/transform/Reshape_1/shapeConst*
valueB"   ????*
_output_shapes
:*
dtype0
?
$bilinear_sampler/transform/Reshape_1Reshape)bilinear_sampler/transform/meshgrid/mul_1*bilinear_sampler/transform/Reshape_1/shape*
T0* 
_output_shapes
:
??*
Tshape0
d
"bilinear_sampler/transform/stack/1Const*
_output_shapes
: *
dtype0*
value	B :
?
 bilinear_sampler/transform/stackPackbilinear_sampler/strided_slice"bilinear_sampler/transform/stack/1*
_output_shapes
:*
N*
T0*

axis 
?
bilinear_sampler/transform/TileTile"bilinear_sampler/transform/Reshape bilinear_sampler/transform/stack*
T0* 
_output_shapes
:
??*

Tmultiples0
f
$bilinear_sampler/transform/stack_1/1Const*
value	B :*
_output_shapes
: *
dtype0
?
"bilinear_sampler/transform/stack_1Packbilinear_sampler/strided_slice$bilinear_sampler/transform/stack_1/1*
T0*
N*

axis *
_output_shapes
:
?
!bilinear_sampler/transform/Tile_1Tile$bilinear_sampler/transform/Reshape_1"bilinear_sampler/transform/stack_1*
T0* 
_output_shapes
:
??*

Tmultiples0
}
*bilinear_sampler/transform/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
$bilinear_sampler/transform/Reshape_2Reshapebilinear_sampler/transform/Tile*bilinear_sampler/transform/Reshape_2/shape*
Tshape0*
T0*
_output_shapes

:??
}
*bilinear_sampler/transform/Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
$bilinear_sampler/transform/Reshape_3Reshape!bilinear_sampler/transform/Tile_1*bilinear_sampler/transform/Reshape_3/shape*
Tshape0*
T0*
_output_shapes

:??
?
.bilinear_sampler/transform/strided_slice/stackConst*
dtype0*%
valueB"                *
_output_shapes
:
?
0bilinear_sampler/transform/strided_slice/stack_1Const*
dtype0*%
valueB"               *
_output_shapes
:
?
0bilinear_sampler/transform/strided_slice/stack_2Const*
dtype0*%
valueB"            *
_output_shapes
:
?
(bilinear_sampler/transform/strided_sliceStridedSlicemul.bilinear_sampler/transform/strided_slice/stack0bilinear_sampler/transform/strided_slice/stack_10bilinear_sampler/transform/strided_slice/stack_2*
T0*
Index0*

begin_mask*(
_output_shapes
:??*
end_mask*
ellipsis_mask *
shrink_axis_mask *
new_axis_mask 
?
0bilinear_sampler/transform/strided_slice_1/stackConst*
dtype0*%
valueB"               *
_output_shapes
:
?
2bilinear_sampler/transform/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*%
valueB"               
?
2bilinear_sampler/transform/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*%
valueB"            
?
*bilinear_sampler/transform/strided_slice_1StridedSlicemul0bilinear_sampler/transform/strided_slice_1/stack2bilinear_sampler/transform/strided_slice_1/stack_12bilinear_sampler/transform/strided_slice_1/stack_2*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *
T0*(
_output_shapes
:??*
Index0*

begin_mask*
end_mask
}
*bilinear_sampler/transform/Reshape_4/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
$bilinear_sampler/transform/Reshape_4Reshape(bilinear_sampler/transform/strided_slice*bilinear_sampler/transform/Reshape_4/shape*
T0*
_output_shapes

:??*
Tshape0
?
bilinear_sampler/transform/mulMul$bilinear_sampler/transform/Reshape_4bilinear_sampler/Cast_1*
_output_shapes

:??*
T0
?
bilinear_sampler/transform/addAdd$bilinear_sampler/transform/Reshape_2bilinear_sampler/transform/mul*
T0*
_output_shapes

:??
}
*bilinear_sampler/transform/Reshape_5/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
?
$bilinear_sampler/transform/Reshape_5Reshape*bilinear_sampler/transform/strided_slice_1*bilinear_sampler/transform/Reshape_5/shape*
T0*
_output_shapes

:??*
Tshape0
?
 bilinear_sampler/transform/mul_1Mul$bilinear_sampler/transform/Reshape_5bilinear_sampler/Cast_1*
T0*
_output_shapes

:??
?
 bilinear_sampler/transform/add_1Add$bilinear_sampler/transform/Reshape_3 bilinear_sampler/transform/mul_1*
T0*
_output_shapes

:??
?
4bilinear_sampler/transform/_interpolate/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
?
+bilinear_sampler/transform/_interpolate/PadPadtruediv4bilinear_sampler/transform/_interpolate/Pad/paddings*
	Tpaddings0*
T0*(
_output_shapes
:??
r
-bilinear_sampler/transform/_interpolate/add/yConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
+bilinear_sampler/transform/_interpolate/addAddbilinear_sampler/transform/add-bilinear_sampler/transform/_interpolate/add/y*
T0*
_output_shapes

:??
t
/bilinear_sampler/transform/_interpolate/add_1/yConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
-bilinear_sampler/transform/_interpolate/add_1Add bilinear_sampler/transform/add_1/bilinear_sampler/transform/_interpolate/add_1/y*
_output_shapes

:??*
T0
r
-bilinear_sampler/transform/_interpolate/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
?
+bilinear_sampler/transform/_interpolate/subSubbilinear_sampler/Cast_1-bilinear_sampler/transform/_interpolate/sub/y*
_output_shapes
: *
T0
t
/bilinear_sampler/transform/_interpolate/add_2/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
?
-bilinear_sampler/transform/_interpolate/add_2Add+bilinear_sampler/transform/_interpolate/sub/bilinear_sampler/transform/_interpolate/add_2/y*
T0*
_output_shapes
: 
?
=bilinear_sampler/transform/_interpolate/clip_by_value/MinimumMinimum+bilinear_sampler/transform/_interpolate/add-bilinear_sampler/transform/_interpolate/add_2*
T0*
_output_shapes

:??
|
7bilinear_sampler/transform/_interpolate/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
5bilinear_sampler/transform/_interpolate/clip_by_valueMaximum=bilinear_sampler/transform/_interpolate/clip_by_value/Minimum7bilinear_sampler/transform/_interpolate/clip_by_value/y*
T0*
_output_shapes

:??
t
/bilinear_sampler/transform/_interpolate/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
?
-bilinear_sampler/transform/_interpolate/sub_1Subbilinear_sampler/Cast/bilinear_sampler/transform/_interpolate/sub_1/y*
T0*
_output_shapes
: 
t
/bilinear_sampler/transform/_interpolate/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
?
-bilinear_sampler/transform/_interpolate/add_3Add-bilinear_sampler/transform/_interpolate/sub_1/bilinear_sampler/transform/_interpolate/add_3/y*
T0*
_output_shapes
: 
?
?bilinear_sampler/transform/_interpolate/clip_by_value_1/MinimumMinimum-bilinear_sampler/transform/_interpolate/add_1-bilinear_sampler/transform/_interpolate/add_3*
T0*
_output_shapes

:??
~
9bilinear_sampler/transform/_interpolate/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
7bilinear_sampler/transform/_interpolate/clip_by_value_1Maximum?bilinear_sampler/transform/_interpolate/clip_by_value_1/Minimum9bilinear_sampler/transform/_interpolate/clip_by_value_1/y*
T0*
_output_shapes

:??
?
-bilinear_sampler/transform/_interpolate/FloorFloor5bilinear_sampler/transform/_interpolate/clip_by_value*
_output_shapes

:??*
T0
?
/bilinear_sampler/transform/_interpolate/Floor_1Floor7bilinear_sampler/transform/_interpolate/clip_by_value_1*
T0*
_output_shapes

:??
t
/bilinear_sampler/transform/_interpolate/add_4/yConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
-bilinear_sampler/transform/_interpolate/add_4Add-bilinear_sampler/transform/_interpolate/Floor/bilinear_sampler/transform/_interpolate/add_4/y*
_output_shapes

:??*
T0
t
/bilinear_sampler/transform/_interpolate/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
-bilinear_sampler/transform/_interpolate/add_5Add/bilinear_sampler/transform/_interpolate/Floor_1/bilinear_sampler/transform/_interpolate/add_5/y*
_output_shapes

:??*
T0
?
,bilinear_sampler/transform/_interpolate/CastCast-bilinear_sampler/transform/_interpolate/Floor*

DstT0*
Truncate( *
_output_shapes

:??*

SrcT0
?
.bilinear_sampler/transform/_interpolate/Cast_1Cast/bilinear_sampler/transform/_interpolate/Floor_1*

SrcT0*
_output_shapes

:??*
Truncate( *

DstT0
t
/bilinear_sampler/transform/_interpolate/sub_2/yConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
-bilinear_sampler/transform/_interpolate/sub_2Subbilinear_sampler/Cast_1/bilinear_sampler/transform/_interpolate/sub_2/y*
T0*
_output_shapes
: 
t
/bilinear_sampler/transform/_interpolate/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
?
-bilinear_sampler/transform/_interpolate/add_6Add-bilinear_sampler/transform/_interpolate/sub_2/bilinear_sampler/transform/_interpolate/add_6/y*
T0*
_output_shapes
: 
?
/bilinear_sampler/transform/_interpolate/MinimumMinimum-bilinear_sampler/transform/_interpolate/add_4-bilinear_sampler/transform/_interpolate/add_6*
_output_shapes

:??*
T0
?
.bilinear_sampler/transform/_interpolate/Cast_2Cast/bilinear_sampler/transform/_interpolate/Minimum*
Truncate( *

DstT0*
_output_shapes

:??*

SrcT0
t
/bilinear_sampler/transform/_interpolate/sub_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
-bilinear_sampler/transform/_interpolate/sub_3Subbilinear_sampler/Cast/bilinear_sampler/transform/_interpolate/sub_3/y*
T0*
_output_shapes
: 
t
/bilinear_sampler/transform/_interpolate/add_7/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
?
-bilinear_sampler/transform/_interpolate/add_7Add-bilinear_sampler/transform/_interpolate/sub_3/bilinear_sampler/transform/_interpolate/add_7/y*
_output_shapes
: *
T0
?
1bilinear_sampler/transform/_interpolate/Minimum_1Minimum-bilinear_sampler/transform/_interpolate/add_5-bilinear_sampler/transform/_interpolate/add_7*
T0*
_output_shapes

:??
?
.bilinear_sampler/transform/_interpolate/Cast_3Cast1bilinear_sampler/transform/_interpolate/Minimum_1*
Truncate( *

DstT0*

SrcT0*
_output_shapes

:??
q
/bilinear_sampler/transform/_interpolate/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :
?
-bilinear_sampler/transform/_interpolate/add_8Add bilinear_sampler/strided_slice_2/bilinear_sampler/transform/_interpolate/add_8/y*
_output_shapes
: *
T0
q
/bilinear_sampler/transform/_interpolate/add_9/yConst*
dtype0*
_output_shapes
: *
value	B :
?
-bilinear_sampler/transform/_interpolate/add_9Add bilinear_sampler/strided_slice_2/bilinear_sampler/transform/_interpolate/add_9/y*
_output_shapes
: *
T0
r
0bilinear_sampler/transform/_interpolate/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :
?
.bilinear_sampler/transform/_interpolate/add_10Add bilinear_sampler/strided_slice_10bilinear_sampler/transform/_interpolate/add_10/y*
T0*
_output_shapes
: 
?
+bilinear_sampler/transform/_interpolate/mulMul-bilinear_sampler/transform/_interpolate/add_9.bilinear_sampler/transform/_interpolate/add_10*
T0*
_output_shapes
: 
u
3bilinear_sampler/transform/_interpolate/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
u
3bilinear_sampler/transform/_interpolate/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
?
-bilinear_sampler/transform/_interpolate/rangeRange3bilinear_sampler/transform/_interpolate/range/startbilinear_sampler/strided_slice3bilinear_sampler/transform/_interpolate/range/delta*
_output_shapes
:*

Tidx0
?
-bilinear_sampler/transform/_interpolate/mul_1Mul-bilinear_sampler/transform/_interpolate/range+bilinear_sampler/transform/_interpolate/mul*
T0*
_output_shapes
:
?
-bilinear_sampler/transform/_interpolate/mul_2Mul bilinear_sampler/strided_slice_1 bilinear_sampler/strided_slice_2*
T0*
_output_shapes
: 
?
>bilinear_sampler/transform/_interpolate/_repeat/ExpandDims/dimConst*
dtype0*
value	B :*
_output_shapes
: 
?
:bilinear_sampler/transform/_interpolate/_repeat/ExpandDims
ExpandDims-bilinear_sampler/transform/_interpolate/mul_1>bilinear_sampler/transform/_interpolate/_repeat/ExpandDims/dim*
T0*
_output_shapes

:*

Tdim0
?
@bilinear_sampler/transform/_interpolate/_repeat/Tile/multiples/0Const*
dtype0*
value	B :*
_output_shapes
: 
?
>bilinear_sampler/transform/_interpolate/_repeat/Tile/multiplesPack@bilinear_sampler/transform/_interpolate/_repeat/Tile/multiples/0-bilinear_sampler/transform/_interpolate/mul_2*
T0*
N*
_output_shapes
:*

axis 
?
4bilinear_sampler/transform/_interpolate/_repeat/TileTile:bilinear_sampler/transform/_interpolate/_repeat/ExpandDims>bilinear_sampler/transform/_interpolate/_repeat/Tile/multiples*

Tmultiples0*
T0* 
_output_shapes
:
??
?
=bilinear_sampler/transform/_interpolate/_repeat/Reshape/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
?
7bilinear_sampler/transform/_interpolate/_repeat/ReshapeReshape4bilinear_sampler/transform/_interpolate/_repeat/Tile=bilinear_sampler/transform/_interpolate/_repeat/Reshape/shape*
_output_shapes

:??*
Tshape0*
T0
?
-bilinear_sampler/transform/_interpolate/mul_3Mul.bilinear_sampler/transform/_interpolate/Cast_1-bilinear_sampler/transform/_interpolate/add_8*
T0*
_output_shapes

:??
?
.bilinear_sampler/transform/_interpolate/add_11Add7bilinear_sampler/transform/_interpolate/_repeat/Reshape-bilinear_sampler/transform/_interpolate/mul_3*
T0*
_output_shapes

:??
?
-bilinear_sampler/transform/_interpolate/mul_4Mul.bilinear_sampler/transform/_interpolate/Cast_3-bilinear_sampler/transform/_interpolate/add_8*
T0*
_output_shapes

:??
?
.bilinear_sampler/transform/_interpolate/add_12Add7bilinear_sampler/transform/_interpolate/_repeat/Reshape-bilinear_sampler/transform/_interpolate/mul_4*
T0*
_output_shapes

:??
?
.bilinear_sampler/transform/_interpolate/add_13Add.bilinear_sampler/transform/_interpolate/add_11,bilinear_sampler/transform/_interpolate/Cast*
T0*
_output_shapes

:??
?
.bilinear_sampler/transform/_interpolate/add_14Add.bilinear_sampler/transform/_interpolate/add_11.bilinear_sampler/transform/_interpolate/Cast_2*
_output_shapes

:??*
T0
?
.bilinear_sampler/transform/_interpolate/add_15Add.bilinear_sampler/transform/_interpolate/add_12,bilinear_sampler/transform/_interpolate/Cast*
T0*
_output_shapes

:??
?
.bilinear_sampler/transform/_interpolate/add_16Add.bilinear_sampler/transform/_interpolate/add_12.bilinear_sampler/transform/_interpolate/Cast_2*
_output_shapes

:??*
T0
z
/bilinear_sampler/transform/_interpolate/stack/0Const*
valueB :
?????????*
_output_shapes
: *
dtype0
?
-bilinear_sampler/transform/_interpolate/stackPack/bilinear_sampler/transform/_interpolate/stack/0 bilinear_sampler/strided_slice_3*
T0*
_output_shapes
:*
N*

axis 
?
/bilinear_sampler/transform/_interpolate/ReshapeReshape+bilinear_sampler/transform/_interpolate/Pad-bilinear_sampler/transform/_interpolate/stack* 
_output_shapes
:
Ƞ*
Tshape0*
T0
w
5bilinear_sampler/transform/_interpolate/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
0bilinear_sampler/transform/_interpolate/GatherV2GatherV2/bilinear_sampler/transform/_interpolate/Reshape.bilinear_sampler/transform/_interpolate/add_135bilinear_sampler/transform/_interpolate/GatherV2/axis* 
_output_shapes
:
??*

batch_dims *
Tparams0*
Tindices0*
Taxis0
y
7bilinear_sampler/transform/_interpolate/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
2bilinear_sampler/transform/_interpolate/GatherV2_1GatherV2/bilinear_sampler/transform/_interpolate/Reshape.bilinear_sampler/transform/_interpolate/add_147bilinear_sampler/transform/_interpolate/GatherV2_1/axis*
Taxis0* 
_output_shapes
:
??*
Tindices0*
Tparams0*

batch_dims 
y
7bilinear_sampler/transform/_interpolate/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
2bilinear_sampler/transform/_interpolate/GatherV2_2GatherV2/bilinear_sampler/transform/_interpolate/Reshape.bilinear_sampler/transform/_interpolate/add_157bilinear_sampler/transform/_interpolate/GatherV2_2/axis*

batch_dims *
Tindices0* 
_output_shapes
:
??*
Tparams0*
Taxis0
y
7bilinear_sampler/transform/_interpolate/GatherV2_3/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
2bilinear_sampler/transform/_interpolate/GatherV2_3GatherV2/bilinear_sampler/transform/_interpolate/Reshape.bilinear_sampler/transform/_interpolate/add_167bilinear_sampler/transform/_interpolate/GatherV2_3/axis*
Taxis0*
Tindices0* 
_output_shapes
:
??*

batch_dims *
Tparams0
?
-bilinear_sampler/transform/_interpolate/sub_4Sub5bilinear_sampler/transform/_interpolate/clip_by_value-bilinear_sampler/transform/_interpolate/Floor*
T0*
_output_shapes

:??
x
6bilinear_sampler/transform/_interpolate/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
?
2bilinear_sampler/transform/_interpolate/ExpandDims
ExpandDims-bilinear_sampler/transform/_interpolate/sub_46bilinear_sampler/transform/_interpolate/ExpandDims/dim*
T0* 
_output_shapes
:
??*

Tdim0
?
-bilinear_sampler/transform/_interpolate/sub_5Sub7bilinear_sampler/transform/_interpolate/clip_by_value_1/bilinear_sampler/transform/_interpolate/Floor_1*
T0*
_output_shapes

:??
z
8bilinear_sampler/transform/_interpolate/ExpandDims_1/dimConst*
dtype0*
value	B :*
_output_shapes
: 
?
4bilinear_sampler/transform/_interpolate/ExpandDims_1
ExpandDims-bilinear_sampler/transform/_interpolate/sub_58bilinear_sampler/transform/_interpolate/ExpandDims_1/dim*
T0* 
_output_shapes
:
??*

Tdim0
?
-bilinear_sampler/transform/_interpolate/sub_6Sub-bilinear_sampler/transform/_interpolate/add_45bilinear_sampler/transform/_interpolate/clip_by_value*
_output_shapes

:??*
T0
z
8bilinear_sampler/transform/_interpolate/ExpandDims_2/dimConst*
dtype0*
value	B :*
_output_shapes
: 
?
4bilinear_sampler/transform/_interpolate/ExpandDims_2
ExpandDims-bilinear_sampler/transform/_interpolate/sub_68bilinear_sampler/transform/_interpolate/ExpandDims_2/dim*
T0* 
_output_shapes
:
??*

Tdim0
?
-bilinear_sampler/transform/_interpolate/sub_7Sub-bilinear_sampler/transform/_interpolate/add_57bilinear_sampler/transform/_interpolate/clip_by_value_1*
T0*
_output_shapes

:??
z
8bilinear_sampler/transform/_interpolate/ExpandDims_3/dimConst*
dtype0*
value	B :*
_output_shapes
: 
?
4bilinear_sampler/transform/_interpolate/ExpandDims_3
ExpandDims-bilinear_sampler/transform/_interpolate/sub_78bilinear_sampler/transform/_interpolate/ExpandDims_3/dim*

Tdim0*
T0* 
_output_shapes
:
??
?
-bilinear_sampler/transform/_interpolate/mul_5Mul0bilinear_sampler/transform/_interpolate/GatherV24bilinear_sampler/transform/_interpolate/ExpandDims_3* 
_output_shapes
:
??*
T0
?
-bilinear_sampler/transform/_interpolate/mul_6Mul-bilinear_sampler/transform/_interpolate/mul_54bilinear_sampler/transform/_interpolate/ExpandDims_2*
T0* 
_output_shapes
:
??
?
-bilinear_sampler/transform/_interpolate/mul_7Mul2bilinear_sampler/transform/_interpolate/GatherV2_14bilinear_sampler/transform/_interpolate/ExpandDims_3*
T0* 
_output_shapes
:
??
?
-bilinear_sampler/transform/_interpolate/mul_8Mul-bilinear_sampler/transform/_interpolate/mul_72bilinear_sampler/transform/_interpolate/ExpandDims*
T0* 
_output_shapes
:
??
?
.bilinear_sampler/transform/_interpolate/add_17Add-bilinear_sampler/transform/_interpolate/mul_6-bilinear_sampler/transform/_interpolate/mul_8*
T0* 
_output_shapes
:
??
?
-bilinear_sampler/transform/_interpolate/mul_9Mul2bilinear_sampler/transform/_interpolate/GatherV2_24bilinear_sampler/transform/_interpolate/ExpandDims_1*
T0* 
_output_shapes
:
??
?
.bilinear_sampler/transform/_interpolate/mul_10Mul-bilinear_sampler/transform/_interpolate/mul_94bilinear_sampler/transform/_interpolate/ExpandDims_2*
T0* 
_output_shapes
:
??
?
.bilinear_sampler/transform/_interpolate/add_18Add.bilinear_sampler/transform/_interpolate/add_17.bilinear_sampler/transform/_interpolate/mul_10*
T0* 
_output_shapes
:
??
?
.bilinear_sampler/transform/_interpolate/mul_11Mul2bilinear_sampler/transform/_interpolate/GatherV2_34bilinear_sampler/transform/_interpolate/ExpandDims_1* 
_output_shapes
:
??*
T0
?
.bilinear_sampler/transform/_interpolate/mul_12Mul.bilinear_sampler/transform/_interpolate/mul_112bilinear_sampler/transform/_interpolate/ExpandDims*
T0* 
_output_shapes
:
??
?
.bilinear_sampler/transform/_interpolate/add_19Add.bilinear_sampler/transform/_interpolate/add_18.bilinear_sampler/transform/_interpolate/mul_12*
T0* 
_output_shapes
:
??
?
"bilinear_sampler/transform/stack_2Packbilinear_sampler/strided_slice bilinear_sampler/strided_slice_1 bilinear_sampler/strided_slice_2 bilinear_sampler/strided_slice_3*
T0*
N*

axis *
_output_shapes
:
?
$bilinear_sampler/transform/Reshape_6Reshape.bilinear_sampler/transform/_interpolate/add_19"bilinear_sampler/transform/stack_2*
T0*(
_output_shapes
:??*
Tshape0
q
bilinear_sampler_1/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?  ?     
p
&bilinear_sampler_1/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
r
(bilinear_sampler_1/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
r
(bilinear_sampler_1/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
 bilinear_sampler_1/strided_sliceStridedSlicebilinear_sampler_1/Shape&bilinear_sampler_1/strided_slice/stack(bilinear_sampler_1/strided_slice/stack_1(bilinear_sampler_1/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *
Index0*
_output_shapes
: *
new_axis_mask *
end_mask *
T0*

begin_mask 
s
bilinear_sampler_1/Shape_1Const*
_output_shapes
:*%
valueB"   ?  ?     *
dtype0
r
(bilinear_sampler_1/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0
t
*bilinear_sampler_1/strided_slice_1/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
t
*bilinear_sampler_1/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
"bilinear_sampler_1/strided_slice_1StridedSlicebilinear_sampler_1/Shape_1(bilinear_sampler_1/strided_slice_1/stack*bilinear_sampler_1/strided_slice_1/stack_1*bilinear_sampler_1/strided_slice_1/stack_2*

begin_mask *
_output_shapes
: *
Index0*
ellipsis_mask *
shrink_axis_mask*
end_mask *
T0*
new_axis_mask 
s
bilinear_sampler_1/Shape_2Const*%
valueB"   ?  ?     *
_output_shapes
:*
dtype0
r
(bilinear_sampler_1/strided_slice_2/stackConst*
valueB:*
_output_shapes
:*
dtype0
t
*bilinear_sampler_1/strided_slice_2/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
t
*bilinear_sampler_1/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
"bilinear_sampler_1/strided_slice_2StridedSlicebilinear_sampler_1/Shape_2(bilinear_sampler_1/strided_slice_2/stack*bilinear_sampler_1/strided_slice_2/stack_1*bilinear_sampler_1/strided_slice_2/stack_2*

begin_mask *
_output_shapes
: *
Index0*
ellipsis_mask *
shrink_axis_mask*
end_mask *
T0*
new_axis_mask 
s
bilinear_sampler_1/Shape_3Const*
dtype0*%
valueB"   ?  ?     *
_output_shapes
:
r
(bilinear_sampler_1/strided_slice_3/stackConst*
valueB:*
_output_shapes
:*
dtype0
t
*bilinear_sampler_1/strided_slice_3/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
t
*bilinear_sampler_1/strided_slice_3/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
"bilinear_sampler_1/strided_slice_3StridedSlicebilinear_sampler_1/Shape_3(bilinear_sampler_1/strided_slice_3/stack*bilinear_sampler_1/strided_slice_3/stack_1*bilinear_sampler_1/strided_slice_3/stack_2*
end_mask *
shrink_axis_mask*
T0*
new_axis_mask *

begin_mask *
ellipsis_mask *
_output_shapes
: *
Index0
?
bilinear_sampler_1/CastCast"bilinear_sampler_1/strided_slice_1*

SrcT0*

DstT0*
_output_shapes
: *
Truncate( 
?
bilinear_sampler_1/Cast_1Cast"bilinear_sampler_1/strided_slice_2*

SrcT0*

DstT0*
_output_shapes
: *
Truncate( 
g
"bilinear_sampler_1/transform/sub/yConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
 bilinear_sampler_1/transform/subSubbilinear_sampler_1/Cast_1"bilinear_sampler_1/transform/sub/y*
T0*
_output_shapes
: 
p
+bilinear_sampler_1/transform/LinSpace/startConst*
valueB
 *    *
_output_shapes
: *
dtype0
?
%bilinear_sampler_1/transform/LinSpaceLinSpace+bilinear_sampler_1/transform/LinSpace/start bilinear_sampler_1/transform/sub"bilinear_sampler_1/strided_slice_2*

Tidx0*
T0*
_output_shapes	
:?
i
$bilinear_sampler_1/transform/sub_1/yConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
"bilinear_sampler_1/transform/sub_1Subbilinear_sampler_1/Cast$bilinear_sampler_1/transform/sub_1/y*
T0*
_output_shapes
: 
r
-bilinear_sampler_1/transform/LinSpace_1/startConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
'bilinear_sampler_1/transform/LinSpace_1LinSpace-bilinear_sampler_1/transform/LinSpace_1/start"bilinear_sampler_1/transform/sub_1"bilinear_sampler_1/strided_slice_1*

Tidx0*
T0*
_output_shapes	
:?
?
3bilinear_sampler_1/transform/meshgrid/Reshape/shapeConst*
dtype0*
valueB"????   *
_output_shapes
:
?
-bilinear_sampler_1/transform/meshgrid/ReshapeReshape%bilinear_sampler_1/transform/LinSpace3bilinear_sampler_1/transform/meshgrid/Reshape/shape*
Tshape0*
T0*
_output_shapes
:	?
?
5bilinear_sampler_1/transform/meshgrid/Reshape_1/shapeConst*
dtype0*
valueB"   ????*
_output_shapes
:
?
/bilinear_sampler_1/transform/meshgrid/Reshape_1Reshape'bilinear_sampler_1/transform/LinSpace_15bilinear_sampler_1/transform/meshgrid/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	?
m
*bilinear_sampler_1/transform/meshgrid/SizeConst*
value
B :?*
_output_shapes
: *
dtype0
o
,bilinear_sampler_1/transform/meshgrid/Size_1Const*
value
B :?*
_output_shapes
: *
dtype0
?
5bilinear_sampler_1/transform/meshgrid/Reshape_2/shapeConst*
dtype0*
valueB"   ????*
_output_shapes
:
?
/bilinear_sampler_1/transform/meshgrid/Reshape_2Reshape-bilinear_sampler_1/transform/meshgrid/Reshape5bilinear_sampler_1/transform/meshgrid/Reshape_2/shape*
Tshape0*
_output_shapes
:	?*
T0
?
5bilinear_sampler_1/transform/meshgrid/Reshape_3/shapeConst*
valueB"????   *
_output_shapes
:*
dtype0
?
/bilinear_sampler_1/transform/meshgrid/Reshape_3Reshape/bilinear_sampler_1/transform/meshgrid/Reshape_15bilinear_sampler_1/transform/meshgrid/Reshape_3/shape*
T0*
Tshape0*
_output_shapes
:	?
?
.bilinear_sampler_1/transform/meshgrid/ones/mulMul,bilinear_sampler_1/transform/meshgrid/Size_1*bilinear_sampler_1/transform/meshgrid/Size*
T0*
_output_shapes
: 
t
1bilinear_sampler_1/transform/meshgrid/ones/Less/yConst*
dtype0*
value
B :?*
_output_shapes
: 
?
/bilinear_sampler_1/transform/meshgrid/ones/LessLess.bilinear_sampler_1/transform/meshgrid/ones/mul1bilinear_sampler_1/transform/meshgrid/ones/Less/y*
T0*
_output_shapes
: 
?
1bilinear_sampler_1/transform/meshgrid/ones/packedPack,bilinear_sampler_1/transform/meshgrid/Size_1*bilinear_sampler_1/transform/meshgrid/Size*

axis *
N*
T0*
_output_shapes
:
u
0bilinear_sampler_1/transform/meshgrid/ones/ConstConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
*bilinear_sampler_1/transform/meshgrid/onesFill1bilinear_sampler_1/transform/meshgrid/ones/packed0bilinear_sampler_1/transform/meshgrid/ones/Const*

index_type0* 
_output_shapes
:
??*
T0
?
)bilinear_sampler_1/transform/meshgrid/mulMul/bilinear_sampler_1/transform/meshgrid/Reshape_2*bilinear_sampler_1/transform/meshgrid/ones* 
_output_shapes
:
??*
T0
?
+bilinear_sampler_1/transform/meshgrid/mul_1Mul/bilinear_sampler_1/transform/meshgrid/Reshape_3*bilinear_sampler_1/transform/meshgrid/ones*
T0* 
_output_shapes
:
??
{
*bilinear_sampler_1/transform/Reshape/shapeConst*
dtype0*
valueB"   ????*
_output_shapes
:
?
$bilinear_sampler_1/transform/ReshapeReshape)bilinear_sampler_1/transform/meshgrid/mul*bilinear_sampler_1/transform/Reshape/shape*
T0*
Tshape0* 
_output_shapes
:
??
}
,bilinear_sampler_1/transform/Reshape_1/shapeConst*
_output_shapes
:*
valueB"   ????*
dtype0
?
&bilinear_sampler_1/transform/Reshape_1Reshape+bilinear_sampler_1/transform/meshgrid/mul_1,bilinear_sampler_1/transform/Reshape_1/shape*
Tshape0* 
_output_shapes
:
??*
T0
f
$bilinear_sampler_1/transform/stack/1Const*
_output_shapes
: *
value	B :*
dtype0
?
"bilinear_sampler_1/transform/stackPack bilinear_sampler_1/strided_slice$bilinear_sampler_1/transform/stack/1*
_output_shapes
:*

axis *
T0*
N
?
!bilinear_sampler_1/transform/TileTile$bilinear_sampler_1/transform/Reshape"bilinear_sampler_1/transform/stack*

Tmultiples0* 
_output_shapes
:
??*
T0
h
&bilinear_sampler_1/transform/stack_1/1Const*
dtype0*
value	B :*
_output_shapes
: 
?
$bilinear_sampler_1/transform/stack_1Pack bilinear_sampler_1/strided_slice&bilinear_sampler_1/transform/stack_1/1*
_output_shapes
:*

axis *
T0*
N
?
#bilinear_sampler_1/transform/Tile_1Tile&bilinear_sampler_1/transform/Reshape_1$bilinear_sampler_1/transform/stack_1*

Tmultiples0* 
_output_shapes
:
??*
T0

,bilinear_sampler_1/transform/Reshape_2/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?
&bilinear_sampler_1/transform/Reshape_2Reshape!bilinear_sampler_1/transform/Tile,bilinear_sampler_1/transform/Reshape_2/shape*
Tshape0*
_output_shapes

:??*
T0

,bilinear_sampler_1/transform/Reshape_3/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
?
&bilinear_sampler_1/transform/Reshape_3Reshape#bilinear_sampler_1/transform/Tile_1,bilinear_sampler_1/transform/Reshape_3/shape*
_output_shapes

:??*
T0*
Tshape0
?
0bilinear_sampler_1/transform/strided_slice/stackConst*
dtype0*%
valueB"                *
_output_shapes
:
?
2bilinear_sampler_1/transform/strided_slice/stack_1Const*
_output_shapes
:*%
valueB"               *
dtype0
?
2bilinear_sampler_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            
?
*bilinear_sampler_1/transform/strided_sliceStridedSlicemul0bilinear_sampler_1/transform/strided_slice/stack2bilinear_sampler_1/transform/strided_slice/stack_12bilinear_sampler_1/transform/strided_slice/stack_2*(
_output_shapes
:??*

begin_mask*
Index0*
shrink_axis_mask *
new_axis_mask *
end_mask*
ellipsis_mask *
T0
?
2bilinear_sampler_1/transform/strided_slice_1/stackConst*
_output_shapes
:*%
valueB"               *
dtype0
?
4bilinear_sampler_1/transform/strided_slice_1/stack_1Const*%
valueB"               *
_output_shapes
:*
dtype0
?
4bilinear_sampler_1/transform/strided_slice_1/stack_2Const*
dtype0*%
valueB"            *
_output_shapes
:
?
,bilinear_sampler_1/transform/strided_slice_1StridedSlicemul2bilinear_sampler_1/transform/strided_slice_1/stack4bilinear_sampler_1/transform/strided_slice_1/stack_14bilinear_sampler_1/transform/strided_slice_1/stack_2*

begin_mask*
Index0*(
_output_shapes
:??*
shrink_axis_mask *
end_mask*
new_axis_mask *
ellipsis_mask *
T0

,bilinear_sampler_1/transform/Reshape_4/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?
&bilinear_sampler_1/transform/Reshape_4Reshape*bilinear_sampler_1/transform/strided_slice,bilinear_sampler_1/transform/Reshape_4/shape*
_output_shapes

:??*
T0*
Tshape0
?
 bilinear_sampler_1/transform/mulMul&bilinear_sampler_1/transform/Reshape_4bilinear_sampler_1/Cast_1*
_output_shapes

:??*
T0
?
 bilinear_sampler_1/transform/addAdd&bilinear_sampler_1/transform/Reshape_2 bilinear_sampler_1/transform/mul*
_output_shapes

:??*
T0

,bilinear_sampler_1/transform/Reshape_5/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?
&bilinear_sampler_1/transform/Reshape_5Reshape,bilinear_sampler_1/transform/strided_slice_1,bilinear_sampler_1/transform/Reshape_5/shape*
_output_shapes

:??*
T0*
Tshape0
?
"bilinear_sampler_1/transform/mul_1Mul&bilinear_sampler_1/transform/Reshape_5bilinear_sampler_1/Cast_1*
_output_shapes

:??*
T0
?
"bilinear_sampler_1/transform/add_1Add&bilinear_sampler_1/transform/Reshape_3"bilinear_sampler_1/transform/mul_1*
_output_shapes

:??*
T0
?
6bilinear_sampler_1/transform/_interpolate/Pad/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
?
-bilinear_sampler_1/transform/_interpolate/PadPadPlaceholder_36bilinear_sampler_1/transform/_interpolate/Pad/paddings*
	Tpaddings0*
T0*(
_output_shapes
:??
t
/bilinear_sampler_1/transform/_interpolate/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
-bilinear_sampler_1/transform/_interpolate/addAdd bilinear_sampler_1/transform/add/bilinear_sampler_1/transform/_interpolate/add/y*
T0*
_output_shapes

:??
v
1bilinear_sampler_1/transform/_interpolate/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
/bilinear_sampler_1/transform/_interpolate/add_1Add"bilinear_sampler_1/transform/add_11bilinear_sampler_1/transform/_interpolate/add_1/y*
T0*
_output_shapes

:??
t
/bilinear_sampler_1/transform/_interpolate/sub/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
?
-bilinear_sampler_1/transform/_interpolate/subSubbilinear_sampler_1/Cast_1/bilinear_sampler_1/transform/_interpolate/sub/y*
T0*
_output_shapes
: 
v
1bilinear_sampler_1/transform/_interpolate/add_2/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
?
/bilinear_sampler_1/transform/_interpolate/add_2Add-bilinear_sampler_1/transform/_interpolate/sub1bilinear_sampler_1/transform/_interpolate/add_2/y*
T0*
_output_shapes
: 
?
?bilinear_sampler_1/transform/_interpolate/clip_by_value/MinimumMinimum-bilinear_sampler_1/transform/_interpolate/add/bilinear_sampler_1/transform/_interpolate/add_2*
_output_shapes

:??*
T0
~
9bilinear_sampler_1/transform/_interpolate/clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
7bilinear_sampler_1/transform/_interpolate/clip_by_valueMaximum?bilinear_sampler_1/transform/_interpolate/clip_by_value/Minimum9bilinear_sampler_1/transform/_interpolate/clip_by_value/y*
_output_shapes

:??*
T0
v
1bilinear_sampler_1/transform/_interpolate/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
?
/bilinear_sampler_1/transform/_interpolate/sub_1Subbilinear_sampler_1/Cast1bilinear_sampler_1/transform/_interpolate/sub_1/y*
_output_shapes
: *
T0
v
1bilinear_sampler_1/transform/_interpolate/add_3/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
?
/bilinear_sampler_1/transform/_interpolate/add_3Add/bilinear_sampler_1/transform/_interpolate/sub_11bilinear_sampler_1/transform/_interpolate/add_3/y*
_output_shapes
: *
T0
?
Abilinear_sampler_1/transform/_interpolate/clip_by_value_1/MinimumMinimum/bilinear_sampler_1/transform/_interpolate/add_1/bilinear_sampler_1/transform/_interpolate/add_3*
_output_shapes

:??*
T0
?
;bilinear_sampler_1/transform/_interpolate/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
9bilinear_sampler_1/transform/_interpolate/clip_by_value_1MaximumAbilinear_sampler_1/transform/_interpolate/clip_by_value_1/Minimum;bilinear_sampler_1/transform/_interpolate/clip_by_value_1/y*
_output_shapes

:??*
T0
?
/bilinear_sampler_1/transform/_interpolate/FloorFloor7bilinear_sampler_1/transform/_interpolate/clip_by_value*
_output_shapes

:??*
T0
?
1bilinear_sampler_1/transform/_interpolate/Floor_1Floor9bilinear_sampler_1/transform/_interpolate/clip_by_value_1*
T0*
_output_shapes

:??
v
1bilinear_sampler_1/transform/_interpolate/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
/bilinear_sampler_1/transform/_interpolate/add_4Add/bilinear_sampler_1/transform/_interpolate/Floor1bilinear_sampler_1/transform/_interpolate/add_4/y*
_output_shapes

:??*
T0
v
1bilinear_sampler_1/transform/_interpolate/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
/bilinear_sampler_1/transform/_interpolate/add_5Add1bilinear_sampler_1/transform/_interpolate/Floor_11bilinear_sampler_1/transform/_interpolate/add_5/y*
_output_shapes

:??*
T0
?
.bilinear_sampler_1/transform/_interpolate/CastCast/bilinear_sampler_1/transform/_interpolate/Floor*
Truncate( *
_output_shapes

:??*

DstT0*

SrcT0
?
0bilinear_sampler_1/transform/_interpolate/Cast_1Cast1bilinear_sampler_1/transform/_interpolate/Floor_1*

DstT0*
_output_shapes

:??*
Truncate( *

SrcT0
v
1bilinear_sampler_1/transform/_interpolate/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
/bilinear_sampler_1/transform/_interpolate/sub_2Subbilinear_sampler_1/Cast_11bilinear_sampler_1/transform/_interpolate/sub_2/y*
_output_shapes
: *
T0
v
1bilinear_sampler_1/transform/_interpolate/add_6/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
?
/bilinear_sampler_1/transform/_interpolate/add_6Add/bilinear_sampler_1/transform/_interpolate/sub_21bilinear_sampler_1/transform/_interpolate/add_6/y*
_output_shapes
: *
T0
?
1bilinear_sampler_1/transform/_interpolate/MinimumMinimum/bilinear_sampler_1/transform/_interpolate/add_4/bilinear_sampler_1/transform/_interpolate/add_6*
_output_shapes

:??*
T0
?
0bilinear_sampler_1/transform/_interpolate/Cast_2Cast1bilinear_sampler_1/transform/_interpolate/Minimum*
Truncate( *
_output_shapes

:??*

DstT0*

SrcT0
v
1bilinear_sampler_1/transform/_interpolate/sub_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
/bilinear_sampler_1/transform/_interpolate/sub_3Subbilinear_sampler_1/Cast1bilinear_sampler_1/transform/_interpolate/sub_3/y*
_output_shapes
: *
T0
v
1bilinear_sampler_1/transform/_interpolate/add_7/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
?
/bilinear_sampler_1/transform/_interpolate/add_7Add/bilinear_sampler_1/transform/_interpolate/sub_31bilinear_sampler_1/transform/_interpolate/add_7/y*
T0*
_output_shapes
: 
?
3bilinear_sampler_1/transform/_interpolate/Minimum_1Minimum/bilinear_sampler_1/transform/_interpolate/add_5/bilinear_sampler_1/transform/_interpolate/add_7*
T0*
_output_shapes

:??
?
0bilinear_sampler_1/transform/_interpolate/Cast_3Cast3bilinear_sampler_1/transform/_interpolate/Minimum_1*
_output_shapes

:??*

DstT0*

SrcT0*
Truncate( 
s
1bilinear_sampler_1/transform/_interpolate/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :
?
/bilinear_sampler_1/transform/_interpolate/add_8Add"bilinear_sampler_1/strided_slice_21bilinear_sampler_1/transform/_interpolate/add_8/y*
_output_shapes
: *
T0
s
1bilinear_sampler_1/transform/_interpolate/add_9/yConst*
dtype0*
_output_shapes
: *
value	B :
?
/bilinear_sampler_1/transform/_interpolate/add_9Add"bilinear_sampler_1/strided_slice_21bilinear_sampler_1/transform/_interpolate/add_9/y*
_output_shapes
: *
T0
t
2bilinear_sampler_1/transform/_interpolate/add_10/yConst*
dtype0*
_output_shapes
: *
value	B :
?
0bilinear_sampler_1/transform/_interpolate/add_10Add"bilinear_sampler_1/strided_slice_12bilinear_sampler_1/transform/_interpolate/add_10/y*
_output_shapes
: *
T0
?
-bilinear_sampler_1/transform/_interpolate/mulMul/bilinear_sampler_1/transform/_interpolate/add_90bilinear_sampler_1/transform/_interpolate/add_10*
_output_shapes
: *
T0
w
5bilinear_sampler_1/transform/_interpolate/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
w
5bilinear_sampler_1/transform/_interpolate/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
/bilinear_sampler_1/transform/_interpolate/rangeRange5bilinear_sampler_1/transform/_interpolate/range/start bilinear_sampler_1/strided_slice5bilinear_sampler_1/transform/_interpolate/range/delta*
_output_shapes
:*

Tidx0
?
/bilinear_sampler_1/transform/_interpolate/mul_1Mul/bilinear_sampler_1/transform/_interpolate/range-bilinear_sampler_1/transform/_interpolate/mul*
_output_shapes
:*
T0
?
/bilinear_sampler_1/transform/_interpolate/mul_2Mul"bilinear_sampler_1/strided_slice_1"bilinear_sampler_1/strided_slice_2*
_output_shapes
: *
T0
?
@bilinear_sampler_1/transform/_interpolate/_repeat/ExpandDims/dimConst*
_output_shapes
: *
value	B :*
dtype0
?
<bilinear_sampler_1/transform/_interpolate/_repeat/ExpandDims
ExpandDims/bilinear_sampler_1/transform/_interpolate/mul_1@bilinear_sampler_1/transform/_interpolate/_repeat/ExpandDims/dim*
_output_shapes

:*
T0*

Tdim0
?
Bbilinear_sampler_1/transform/_interpolate/_repeat/Tile/multiples/0Const*
value	B :*
_output_shapes
: *
dtype0
?
@bilinear_sampler_1/transform/_interpolate/_repeat/Tile/multiplesPackBbilinear_sampler_1/transform/_interpolate/_repeat/Tile/multiples/0/bilinear_sampler_1/transform/_interpolate/mul_2*
_output_shapes
:*
T0*

axis *
N
?
6bilinear_sampler_1/transform/_interpolate/_repeat/TileTile<bilinear_sampler_1/transform/_interpolate/_repeat/ExpandDims@bilinear_sampler_1/transform/_interpolate/_repeat/Tile/multiples* 
_output_shapes
:
??*
T0*

Tmultiples0
?
?bilinear_sampler_1/transform/_interpolate/_repeat/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
9bilinear_sampler_1/transform/_interpolate/_repeat/ReshapeReshape6bilinear_sampler_1/transform/_interpolate/_repeat/Tile?bilinear_sampler_1/transform/_interpolate/_repeat/Reshape/shape*
_output_shapes

:??*
T0*
Tshape0
?
/bilinear_sampler_1/transform/_interpolate/mul_3Mul0bilinear_sampler_1/transform/_interpolate/Cast_1/bilinear_sampler_1/transform/_interpolate/add_8*
T0*
_output_shapes

:??
?
0bilinear_sampler_1/transform/_interpolate/add_11Add9bilinear_sampler_1/transform/_interpolate/_repeat/Reshape/bilinear_sampler_1/transform/_interpolate/mul_3*
T0*
_output_shapes

:??
?
/bilinear_sampler_1/transform/_interpolate/mul_4Mul0bilinear_sampler_1/transform/_interpolate/Cast_3/bilinear_sampler_1/transform/_interpolate/add_8*
_output_shapes

:??*
T0
?
0bilinear_sampler_1/transform/_interpolate/add_12Add9bilinear_sampler_1/transform/_interpolate/_repeat/Reshape/bilinear_sampler_1/transform/_interpolate/mul_4*
_output_shapes

:??*
T0
?
0bilinear_sampler_1/transform/_interpolate/add_13Add0bilinear_sampler_1/transform/_interpolate/add_11.bilinear_sampler_1/transform/_interpolate/Cast*
_output_shapes

:??*
T0
?
0bilinear_sampler_1/transform/_interpolate/add_14Add0bilinear_sampler_1/transform/_interpolate/add_110bilinear_sampler_1/transform/_interpolate/Cast_2*
_output_shapes

:??*
T0
?
0bilinear_sampler_1/transform/_interpolate/add_15Add0bilinear_sampler_1/transform/_interpolate/add_12.bilinear_sampler_1/transform/_interpolate/Cast*
_output_shapes

:??*
T0
?
0bilinear_sampler_1/transform/_interpolate/add_16Add0bilinear_sampler_1/transform/_interpolate/add_120bilinear_sampler_1/transform/_interpolate/Cast_2*
_output_shapes

:??*
T0
|
1bilinear_sampler_1/transform/_interpolate/stack/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
/bilinear_sampler_1/transform/_interpolate/stackPack1bilinear_sampler_1/transform/_interpolate/stack/0"bilinear_sampler_1/strided_slice_3*
_output_shapes
:*
T0*

axis *
N
?
1bilinear_sampler_1/transform/_interpolate/ReshapeReshape-bilinear_sampler_1/transform/_interpolate/Pad/bilinear_sampler_1/transform/_interpolate/stack* 
_output_shapes
:
Ƞ*
T0*
Tshape0
y
7bilinear_sampler_1/transform/_interpolate/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
2bilinear_sampler_1/transform/_interpolate/GatherV2GatherV21bilinear_sampler_1/transform/_interpolate/Reshape0bilinear_sampler_1/transform/_interpolate/add_137bilinear_sampler_1/transform/_interpolate/GatherV2/axis*

batch_dims * 
_output_shapes
:
??*
Tindices0*
Tparams0*
Taxis0
{
9bilinear_sampler_1/transform/_interpolate/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
4bilinear_sampler_1/transform/_interpolate/GatherV2_1GatherV21bilinear_sampler_1/transform/_interpolate/Reshape0bilinear_sampler_1/transform/_interpolate/add_149bilinear_sampler_1/transform/_interpolate/GatherV2_1/axis*
Tindices0* 
_output_shapes
:
??*
Tparams0*
Taxis0*

batch_dims 
{
9bilinear_sampler_1/transform/_interpolate/GatherV2_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
4bilinear_sampler_1/transform/_interpolate/GatherV2_2GatherV21bilinear_sampler_1/transform/_interpolate/Reshape0bilinear_sampler_1/transform/_interpolate/add_159bilinear_sampler_1/transform/_interpolate/GatherV2_2/axis* 
_output_shapes
:
??*
Taxis0*

batch_dims *
Tindices0*
Tparams0
{
9bilinear_sampler_1/transform/_interpolate/GatherV2_3/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
4bilinear_sampler_1/transform/_interpolate/GatherV2_3GatherV21bilinear_sampler_1/transform/_interpolate/Reshape0bilinear_sampler_1/transform/_interpolate/add_169bilinear_sampler_1/transform/_interpolate/GatherV2_3/axis*
Taxis0*
Tparams0*
Tindices0*

batch_dims * 
_output_shapes
:
??
?
/bilinear_sampler_1/transform/_interpolate/sub_4Sub7bilinear_sampler_1/transform/_interpolate/clip_by_value/bilinear_sampler_1/transform/_interpolate/Floor*
T0*
_output_shapes

:??
z
8bilinear_sampler_1/transform/_interpolate/ExpandDims/dimConst*
_output_shapes
: *
value	B :*
dtype0
?
4bilinear_sampler_1/transform/_interpolate/ExpandDims
ExpandDims/bilinear_sampler_1/transform/_interpolate/sub_48bilinear_sampler_1/transform/_interpolate/ExpandDims/dim*
T0*

Tdim0* 
_output_shapes
:
??
?
/bilinear_sampler_1/transform/_interpolate/sub_5Sub9bilinear_sampler_1/transform/_interpolate/clip_by_value_11bilinear_sampler_1/transform/_interpolate/Floor_1*
_output_shapes

:??*
T0
|
:bilinear_sampler_1/transform/_interpolate/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
?
6bilinear_sampler_1/transform/_interpolate/ExpandDims_1
ExpandDims/bilinear_sampler_1/transform/_interpolate/sub_5:bilinear_sampler_1/transform/_interpolate/ExpandDims_1/dim*
T0*

Tdim0* 
_output_shapes
:
??
?
/bilinear_sampler_1/transform/_interpolate/sub_6Sub/bilinear_sampler_1/transform/_interpolate/add_47bilinear_sampler_1/transform/_interpolate/clip_by_value*
_output_shapes

:??*
T0
|
:bilinear_sampler_1/transform/_interpolate/ExpandDims_2/dimConst*
_output_shapes
: *
value	B :*
dtype0
?
6bilinear_sampler_1/transform/_interpolate/ExpandDims_2
ExpandDims/bilinear_sampler_1/transform/_interpolate/sub_6:bilinear_sampler_1/transform/_interpolate/ExpandDims_2/dim*

Tdim0*
T0* 
_output_shapes
:
??
?
/bilinear_sampler_1/transform/_interpolate/sub_7Sub/bilinear_sampler_1/transform/_interpolate/add_59bilinear_sampler_1/transform/_interpolate/clip_by_value_1*
_output_shapes

:??*
T0
|
:bilinear_sampler_1/transform/_interpolate/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :
?
6bilinear_sampler_1/transform/_interpolate/ExpandDims_3
ExpandDims/bilinear_sampler_1/transform/_interpolate/sub_7:bilinear_sampler_1/transform/_interpolate/ExpandDims_3/dim* 
_output_shapes
:
??*

Tdim0*
T0
?
/bilinear_sampler_1/transform/_interpolate/mul_5Mul2bilinear_sampler_1/transform/_interpolate/GatherV26bilinear_sampler_1/transform/_interpolate/ExpandDims_3* 
_output_shapes
:
??*
T0
?
/bilinear_sampler_1/transform/_interpolate/mul_6Mul/bilinear_sampler_1/transform/_interpolate/mul_56bilinear_sampler_1/transform/_interpolate/ExpandDims_2* 
_output_shapes
:
??*
T0
?
/bilinear_sampler_1/transform/_interpolate/mul_7Mul4bilinear_sampler_1/transform/_interpolate/GatherV2_16bilinear_sampler_1/transform/_interpolate/ExpandDims_3* 
_output_shapes
:
??*
T0
?
/bilinear_sampler_1/transform/_interpolate/mul_8Mul/bilinear_sampler_1/transform/_interpolate/mul_74bilinear_sampler_1/transform/_interpolate/ExpandDims* 
_output_shapes
:
??*
T0
?
0bilinear_sampler_1/transform/_interpolate/add_17Add/bilinear_sampler_1/transform/_interpolate/mul_6/bilinear_sampler_1/transform/_interpolate/mul_8* 
_output_shapes
:
??*
T0
?
/bilinear_sampler_1/transform/_interpolate/mul_9Mul4bilinear_sampler_1/transform/_interpolate/GatherV2_26bilinear_sampler_1/transform/_interpolate/ExpandDims_1* 
_output_shapes
:
??*
T0
?
0bilinear_sampler_1/transform/_interpolate/mul_10Mul/bilinear_sampler_1/transform/_interpolate/mul_96bilinear_sampler_1/transform/_interpolate/ExpandDims_2* 
_output_shapes
:
??*
T0
?
0bilinear_sampler_1/transform/_interpolate/add_18Add0bilinear_sampler_1/transform/_interpolate/add_170bilinear_sampler_1/transform/_interpolate/mul_10* 
_output_shapes
:
??*
T0
?
0bilinear_sampler_1/transform/_interpolate/mul_11Mul4bilinear_sampler_1/transform/_interpolate/GatherV2_36bilinear_sampler_1/transform/_interpolate/ExpandDims_1* 
_output_shapes
:
??*
T0
?
0bilinear_sampler_1/transform/_interpolate/mul_12Mul0bilinear_sampler_1/transform/_interpolate/mul_114bilinear_sampler_1/transform/_interpolate/ExpandDims* 
_output_shapes
:
??*
T0
?
0bilinear_sampler_1/transform/_interpolate/add_19Add0bilinear_sampler_1/transform/_interpolate/add_180bilinear_sampler_1/transform/_interpolate/mul_12* 
_output_shapes
:
??*
T0
?
$bilinear_sampler_1/transform/stack_2Pack bilinear_sampler_1/strided_slice"bilinear_sampler_1/strided_slice_1"bilinear_sampler_1/strided_slice_2"bilinear_sampler_1/strided_slice_3*
_output_shapes
:*
T0*
N*

axis 
?
&bilinear_sampler_1/transform/Reshape_6Reshape0bilinear_sampler_1/transform/_interpolate/add_19$bilinear_sampler_1/transform/stack_2*
Tshape0*(
_output_shapes
:??*
T0
l
mul_1Mul$bilinear_sampler/transform/Reshape_6Const*(
_output_shapes
:??*
T0
m
mul_2MulTile&bilinear_sampler_1/transform/Reshape_6*(
_output_shapes
:??*
T0
K
sub_1Submul_2mul*(
_output_shapes
:??*
T0
D
AbsAbssub_1*(
_output_shapes
:??*
T0
`
Sum/reduction_indicesConst*
_output_shapes
: *
valueB :
?????????*
dtype0
v
SumSumAbsSum/reduction_indices*(
_output_shapes
:??*
	keep_dims(*
T0*

Tidx0
L
mul_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  |?
M
mul_3Mulmul_3/xSum*(
_output_shapes
:??*
T0
D
ExpExpmul_3*(
_output_shapes
:??*
T0
Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
value	B : *
dtype0
z
Sum_1SumExpSum_1/reduction_indices*(
_output_shapes
:??*
T0*
	keep_dims(*

Tidx0
J
add/yConst*
_output_shapes
: *
valueB
 *??'7*
dtype0
K
addAddSum_1add/y*(
_output_shapes
:??*
T0
Q
	truediv_1RealDivExpadd*(
_output_shapes
:??*
T0
Q
Mul_4Mulmul_1	truediv_1*(
_output_shapes
:??*
T0
Y
Sum_2/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B : 
|
Sum_2SumMul_4Sum_2/reduction_indices*(
_output_shapes
:??*
	keep_dims(*
T0*

Tidx0
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
_output_shapes
: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
?
save/SaveV2/tensor_namesConst*
_output_shapes
:*
dtype0*?
value?B?Bgen_flows/layer_0/conv2d/biasBgen_flows/layer_0/conv2d/kernelBgen_flows/layer_1/conv2d/biasBgen_flows/layer_1/conv2d/kernelBgen_flows/layer_2/conv2d/biasBgen_flows/layer_2/conv2d/kernelBgen_flows/layer_3/conv2d/biasBgen_flows/layer_3/conv2d/kernelBgen_flows/layer_4/conv2d/biasBgen_flows/layer_4/conv2d/kernelBgen_flows/layer_5/conv2d/biasBgen_flows/layer_5/conv2d/kernelBgen_flows/layer_6/conv2d/biasBgen_flows/layer_6/conv2d/kernelBgen_flows/layer_7/conv2d/biasBgen_flows/layer_7/conv2d/kernelB#gen_flows/outputs_flows/conv2d/biasB%gen_flows/outputs_flows/conv2d/kernel
?
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesgen_flows/layer_0/conv2d/biasgen_flows/layer_0/conv2d/kernelgen_flows/layer_1/conv2d/biasgen_flows/layer_1/conv2d/kernelgen_flows/layer_2/conv2d/biasgen_flows/layer_2/conv2d/kernelgen_flows/layer_3/conv2d/biasgen_flows/layer_3/conv2d/kernelgen_flows/layer_4/conv2d/biasgen_flows/layer_4/conv2d/kernelgen_flows/layer_5/conv2d/biasgen_flows/layer_5/conv2d/kernelgen_flows/layer_6/conv2d/biasgen_flows/layer_6/conv2d/kernelgen_flows/layer_7/conv2d/biasgen_flows/layer_7/conv2d/kernel#gen_flows/outputs_flows/conv2d/bias%gen_flows/outputs_flows/conv2d/kernel* 
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
_class
loc:@save/Const*
T0
?
save/RestoreV2/tensor_namesConst*
_output_shapes
:*?
value?B?Bgen_flows/layer_0/conv2d/biasBgen_flows/layer_0/conv2d/kernelBgen_flows/layer_1/conv2d/biasBgen_flows/layer_1/conv2d/kernelBgen_flows/layer_2/conv2d/biasBgen_flows/layer_2/conv2d/kernelBgen_flows/layer_3/conv2d/biasBgen_flows/layer_3/conv2d/kernelBgen_flows/layer_4/conv2d/biasBgen_flows/layer_4/conv2d/kernelBgen_flows/layer_5/conv2d/biasBgen_flows/layer_5/conv2d/kernelBgen_flows/layer_6/conv2d/biasBgen_flows/layer_6/conv2d/kernelBgen_flows/layer_7/conv2d/biasBgen_flows/layer_7/conv2d/kernelB#gen_flows/outputs_flows/conv2d/biasB%gen_flows/outputs_flows/conv2d/kernel*
dtype0
?
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*7
value.B,B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2
?
save/AssignAssigngen_flows/layer_0/conv2d/biassave/RestoreV2*
use_locking(*0
_class&
$"loc:@gen_flows/layer_0/conv2d/bias*
_output_shapes	
:?*
T0*
validate_shape(
?
save/Assign_1Assigngen_flows/layer_0/conv2d/kernelsave/RestoreV2:1*'
_output_shapes
:?*
validate_shape(*2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel*
T0*
use_locking(
?
save/Assign_2Assigngen_flows/layer_1/conv2d/biassave/RestoreV2:2*0
_class&
$"loc:@gen_flows/layer_1/conv2d/bias*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(
?
save/Assign_3Assigngen_flows/layer_1/conv2d/kernelsave/RestoreV2:3*2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel*
use_locking(*
validate_shape(*(
_output_shapes
:??*
T0
?
save/Assign_4Assigngen_flows/layer_2/conv2d/biassave/RestoreV2:4*
use_locking(*0
_class&
$"loc:@gen_flows/layer_2/conv2d/bias*
_output_shapes	
:?*
T0*
validate_shape(
?
save/Assign_5Assigngen_flows/layer_2/conv2d/kernelsave/RestoreV2:5*
use_locking(*
validate_shape(*(
_output_shapes
:??*
T0*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel
?
save/Assign_6Assigngen_flows/layer_3/conv2d/biassave/RestoreV2:6*0
_class&
$"loc:@gen_flows/layer_3/conv2d/bias*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0
?
save/Assign_7Assigngen_flows/layer_3/conv2d/kernelsave/RestoreV2:7*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel*
use_locking(*
validate_shape(*'
_output_shapes
:?@*
T0
?
save/Assign_8Assigngen_flows/layer_4/conv2d/biassave/RestoreV2:8*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0*0
_class&
$"loc:@gen_flows/layer_4/conv2d/bias
?
save/Assign_9Assigngen_flows/layer_4/conv2d/kernelsave/RestoreV2:9*2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel*
use_locking(*
validate_shape(*&
_output_shapes
:@@*
T0
?
save/Assign_10Assigngen_flows/layer_5/conv2d/biassave/RestoreV2:10*
use_locking(*0
_class&
$"loc:@gen_flows/layer_5/conv2d/bias*
_output_shapes
:@*
T0*
validate_shape(
?
save/Assign_11Assigngen_flows/layer_5/conv2d/kernelsave/RestoreV2:11*&
_output_shapes
:@@*
validate_shape(*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*
T0*
use_locking(
?
save/Assign_12Assigngen_flows/layer_6/conv2d/biassave/RestoreV2:12*
_output_shapes
: *0
_class&
$"loc:@gen_flows/layer_6/conv2d/bias*
validate_shape(*
T0*
use_locking(
?
save/Assign_13Assigngen_flows/layer_6/conv2d/kernelsave/RestoreV2:13*
use_locking(*
validate_shape(*&
_output_shapes
:@ *
T0*2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel
?
save/Assign_14Assigngen_flows/layer_7/conv2d/biassave/RestoreV2:14*
_output_shapes
: *0
_class&
$"loc:@gen_flows/layer_7/conv2d/bias*
validate_shape(*
T0*
use_locking(
?
save/Assign_15Assigngen_flows/layer_7/conv2d/kernelsave/RestoreV2:15*2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel*
use_locking(*
validate_shape(*&
_output_shapes
:  *
T0
?
save/Assign_16Assign#gen_flows/outputs_flows/conv2d/biassave/RestoreV2:16*
validate_shape(*
_output_shapes
:*
T0*6
_class,
*(loc:@gen_flows/outputs_flows/conv2d/bias*
use_locking(
?
save/Assign_17Assign%gen_flows/outputs_flows/conv2d/kernelsave/RestoreV2:17*
validate_shape(*&
_output_shapes
: *
T0*8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel*
use_locking(
?
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
?
initNoOp%^gen_flows/layer_0/conv2d/bias/Assign'^gen_flows/layer_0/conv2d/kernel/Assign%^gen_flows/layer_1/conv2d/bias/Assign'^gen_flows/layer_1/conv2d/kernel/Assign%^gen_flows/layer_2/conv2d/bias/Assign'^gen_flows/layer_2/conv2d/kernel/Assign%^gen_flows/layer_3/conv2d/bias/Assign'^gen_flows/layer_3/conv2d/kernel/Assign%^gen_flows/layer_4/conv2d/bias/Assign'^gen_flows/layer_4/conv2d/kernel/Assign%^gen_flows/layer_5/conv2d/bias/Assign'^gen_flows/layer_5/conv2d/kernel/Assign%^gen_flows/layer_6/conv2d/bias/Assign'^gen_flows/layer_6/conv2d/kernel/Assign%^gen_flows/layer_7/conv2d/bias/Assign'^gen_flows/layer_7/conv2d/kernel/Assign+^gen_flows/outputs_flows/conv2d/bias/Assign-^gen_flows/outputs_flows/conv2d/kernel/Assign
[
save_1/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 
?
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_d007780a7bc248b2b14eaa9cfca9219c/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
?
save_1/SaveV2/tensor_namesConst*?
value?B?Bgen_flows/layer_0/conv2d/biasBgen_flows/layer_0/conv2d/kernelBgen_flows/layer_1/conv2d/biasBgen_flows/layer_1/conv2d/kernelBgen_flows/layer_2/conv2d/biasBgen_flows/layer_2/conv2d/kernelBgen_flows/layer_3/conv2d/biasBgen_flows/layer_3/conv2d/kernelBgen_flows/layer_4/conv2d/biasBgen_flows/layer_4/conv2d/kernelBgen_flows/layer_5/conv2d/biasBgen_flows/layer_5/conv2d/kernelBgen_flows/layer_6/conv2d/biasBgen_flows/layer_6/conv2d/kernelBgen_flows/layer_7/conv2d/biasBgen_flows/layer_7/conv2d/kernelB#gen_flows/outputs_flows/conv2d/biasB%gen_flows/outputs_flows/conv2d/kernel*
_output_shapes
:*
dtype0
?
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesgen_flows/layer_0/conv2d/biasgen_flows/layer_0/conv2d/kernelgen_flows/layer_1/conv2d/biasgen_flows/layer_1/conv2d/kernelgen_flows/layer_2/conv2d/biasgen_flows/layer_2/conv2d/kernelgen_flows/layer_3/conv2d/biasgen_flows/layer_3/conv2d/kernelgen_flows/layer_4/conv2d/biasgen_flows/layer_4/conv2d/kernelgen_flows/layer_5/conv2d/biasgen_flows/layer_5/conv2d/kernelgen_flows/layer_6/conv2d/biasgen_flows/layer_6/conv2d/kernelgen_flows/layer_7/conv2d/biasgen_flows/layer_7/conv2d/kernel#gen_flows/outputs_flows/conv2d/bias%gen_flows/outputs_flows/conv2d/kernel* 
dtypes
2
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: *
T0
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
N*
T0*
_output_shapes
:
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
_output_shapes
: *
T0
?
save_1/RestoreV2/tensor_namesConst*?
value?B?Bgen_flows/layer_0/conv2d/biasBgen_flows/layer_0/conv2d/kernelBgen_flows/layer_1/conv2d/biasBgen_flows/layer_1/conv2d/kernelBgen_flows/layer_2/conv2d/biasBgen_flows/layer_2/conv2d/kernelBgen_flows/layer_3/conv2d/biasBgen_flows/layer_3/conv2d/kernelBgen_flows/layer_4/conv2d/biasBgen_flows/layer_4/conv2d/kernelBgen_flows/layer_5/conv2d/biasBgen_flows/layer_5/conv2d/kernelBgen_flows/layer_6/conv2d/biasBgen_flows/layer_6/conv2d/kernelBgen_flows/layer_7/conv2d/biasBgen_flows/layer_7/conv2d/kernelB#gen_flows/outputs_flows/conv2d/biasB%gen_flows/outputs_flows/conv2d/kernel*
_output_shapes
:*
dtype0
?
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:*7
value.B,B B B B B B B B B B B B B B B B B B *
dtype0
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2
?
save_1/AssignAssigngen_flows/layer_0/conv2d/biassave_1/RestoreV2*
_output_shapes	
:?*
use_locking(*
validate_shape(*0
_class&
$"loc:@gen_flows/layer_0/conv2d/bias*
T0
?
save_1/Assign_1Assigngen_flows/layer_0/conv2d/kernelsave_1/RestoreV2:1*'
_output_shapes
:?*
use_locking(*
validate_shape(*2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel*
T0
?
save_1/Assign_2Assigngen_flows/layer_1/conv2d/biassave_1/RestoreV2:2*0
_class&
$"loc:@gen_flows/layer_1/conv2d/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0
?
save_1/Assign_3Assigngen_flows/layer_1/conv2d/kernelsave_1/RestoreV2:3*
validate_shape(*
T0*(
_output_shapes
:??*
use_locking(*2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel
?
save_1/Assign_4Assigngen_flows/layer_2/conv2d/biassave_1/RestoreV2:4*
validate_shape(*
_output_shapes	
:?*
use_locking(*0
_class&
$"loc:@gen_flows/layer_2/conv2d/bias*
T0
?
save_1/Assign_5Assigngen_flows/layer_2/conv2d/kernelsave_1/RestoreV2:5*(
_output_shapes
:??*
use_locking(*
validate_shape(*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel*
T0
?
save_1/Assign_6Assigngen_flows/layer_3/conv2d/biassave_1/RestoreV2:6*0
_class&
$"loc:@gen_flows/layer_3/conv2d/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
?
save_1/Assign_7Assigngen_flows/layer_3/conv2d/kernelsave_1/RestoreV2:7*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel*
T0*'
_output_shapes
:?@*
use_locking(*
validate_shape(
?
save_1/Assign_8Assigngen_flows/layer_4/conv2d/biassave_1/RestoreV2:8*
validate_shape(*
_output_shapes
:@*
use_locking(*0
_class&
$"loc:@gen_flows/layer_4/conv2d/bias*
T0
?
save_1/Assign_9Assigngen_flows/layer_4/conv2d/kernelsave_1/RestoreV2:9*2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel*&
_output_shapes
:@@*
use_locking(*
validate_shape(*
T0
?
save_1/Assign_10Assigngen_flows/layer_5/conv2d/biassave_1/RestoreV2:10*0
_class&
$"loc:@gen_flows/layer_5/conv2d/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
?
save_1/Assign_11Assigngen_flows/layer_5/conv2d/kernelsave_1/RestoreV2:11*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(
?
save_1/Assign_12Assigngen_flows/layer_6/conv2d/biassave_1/RestoreV2:12*
_output_shapes
: *
use_locking(*
validate_shape(*0
_class&
$"loc:@gen_flows/layer_6/conv2d/bias*
T0
?
save_1/Assign_13Assigngen_flows/layer_6/conv2d/kernelsave_1/RestoreV2:13*2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel*&
_output_shapes
:@ *
use_locking(*
validate_shape(*
T0
?
save_1/Assign_14Assigngen_flows/layer_7/conv2d/biassave_1/RestoreV2:14*0
_class&
$"loc:@gen_flows/layer_7/conv2d/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
?
save_1/Assign_15Assigngen_flows/layer_7/conv2d/kernelsave_1/RestoreV2:15*
validate_shape(*&
_output_shapes
:  *
use_locking(*2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel*
T0
?
save_1/Assign_16Assign#gen_flows/outputs_flows/conv2d/biassave_1/RestoreV2:16*6
_class,
*(loc:@gen_flows/outputs_flows/conv2d/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
?
save_1/Assign_17Assign%gen_flows/outputs_flows/conv2d/kernelsave_1/RestoreV2:17*8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel*&
_output_shapes
: *
use_locking(*
T0*
validate_shape(
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
shape: *
_output_shapes
: *
dtype0
?
save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_84245f3e945947f1b41201c98d45588c/part*
_output_shapes
: *
dtype0
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
?
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
?
save_2/SaveV2/tensor_namesConst*?
value?B?Bgen_flows/layer_0/conv2d/biasBgen_flows/layer_0/conv2d/kernelBgen_flows/layer_1/conv2d/biasBgen_flows/layer_1/conv2d/kernelBgen_flows/layer_2/conv2d/biasBgen_flows/layer_2/conv2d/kernelBgen_flows/layer_3/conv2d/biasBgen_flows/layer_3/conv2d/kernelBgen_flows/layer_4/conv2d/biasBgen_flows/layer_4/conv2d/kernelBgen_flows/layer_5/conv2d/biasBgen_flows/layer_5/conv2d/kernelBgen_flows/layer_6/conv2d/biasBgen_flows/layer_6/conv2d/kernelBgen_flows/layer_7/conv2d/biasBgen_flows/layer_7/conv2d/kernelB#gen_flows/outputs_flows/conv2d/biasB%gen_flows/outputs_flows/conv2d/kernel*
_output_shapes
:*
dtype0
?
save_2/SaveV2/shape_and_slicesConst*
_output_shapes
:*7
value.B,B B B B B B B B B B B B B B B B B B *
dtype0
?
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesgen_flows/layer_0/conv2d/biasgen_flows/layer_0/conv2d/kernelgen_flows/layer_1/conv2d/biasgen_flows/layer_1/conv2d/kernelgen_flows/layer_2/conv2d/biasgen_flows/layer_2/conv2d/kernelgen_flows/layer_3/conv2d/biasgen_flows/layer_3/conv2d/kernelgen_flows/layer_4/conv2d/biasgen_flows/layer_4/conv2d/kernelgen_flows/layer_5/conv2d/biasgen_flows/layer_5/conv2d/kernelgen_flows/layer_6/conv2d/biasgen_flows/layer_6/conv2d/kernelgen_flows/layer_7/conv2d/biasgen_flows/layer_7/conv2d/kernel#gen_flows/outputs_flows/conv2d/bias%gen_flows/outputs_flows/conv2d/kernel* 
dtypes
2
?
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_2/ShardedFilename
?
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
_output_shapes
:*
N*

axis *
T0
?
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
?
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
?
save_2/RestoreV2/tensor_namesConst*?
value?B?Bgen_flows/layer_0/conv2d/biasBgen_flows/layer_0/conv2d/kernelBgen_flows/layer_1/conv2d/biasBgen_flows/layer_1/conv2d/kernelBgen_flows/layer_2/conv2d/biasBgen_flows/layer_2/conv2d/kernelBgen_flows/layer_3/conv2d/biasBgen_flows/layer_3/conv2d/kernelBgen_flows/layer_4/conv2d/biasBgen_flows/layer_4/conv2d/kernelBgen_flows/layer_5/conv2d/biasBgen_flows/layer_5/conv2d/kernelBgen_flows/layer_6/conv2d/biasBgen_flows/layer_6/conv2d/kernelBgen_flows/layer_7/conv2d/biasBgen_flows/layer_7/conv2d/kernelB#gen_flows/outputs_flows/conv2d/biasB%gen_flows/outputs_flows/conv2d/kernel*
_output_shapes
:*
dtype0
?
!save_2/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*7
value.B,B B B B B B B B B B B B B B B B B B 
?
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2
?
save_2/AssignAssigngen_flows/layer_0/conv2d/biassave_2/RestoreV2*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(*0
_class&
$"loc:@gen_flows/layer_0/conv2d/bias
?
save_2/Assign_1Assigngen_flows/layer_0/conv2d/kernelsave_2/RestoreV2:1*'
_output_shapes
:?*
T0*
use_locking(*
validate_shape(*2
_class(
&$loc:@gen_flows/layer_0/conv2d/kernel
?
save_2/Assign_2Assigngen_flows/layer_1/conv2d/biassave_2/RestoreV2:2*
_output_shapes	
:?*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@gen_flows/layer_1/conv2d/bias
?
save_2/Assign_3Assigngen_flows/layer_1/conv2d/kernelsave_2/RestoreV2:3*2
_class(
&$loc:@gen_flows/layer_1/conv2d/kernel*(
_output_shapes
:??*
validate_shape(*
T0*
use_locking(
?
save_2/Assign_4Assigngen_flows/layer_2/conv2d/biassave_2/RestoreV2:4*
use_locking(*0
_class&
$"loc:@gen_flows/layer_2/conv2d/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
save_2/Assign_5Assigngen_flows/layer_2/conv2d/kernelsave_2/RestoreV2:5*
T0*
use_locking(*(
_output_shapes
:??*2
_class(
&$loc:@gen_flows/layer_2/conv2d/kernel*
validate_shape(
?
save_2/Assign_6Assigngen_flows/layer_3/conv2d/biassave_2/RestoreV2:6*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(*0
_class&
$"loc:@gen_flows/layer_3/conv2d/bias
?
save_2/Assign_7Assigngen_flows/layer_3/conv2d/kernelsave_2/RestoreV2:7*
validate_shape(*
use_locking(*
T0*2
_class(
&$loc:@gen_flows/layer_3/conv2d/kernel*'
_output_shapes
:?@
?
save_2/Assign_8Assigngen_flows/layer_4/conv2d/biassave_2/RestoreV2:8*
validate_shape(*
T0*
use_locking(*0
_class&
$"loc:@gen_flows/layer_4/conv2d/bias*
_output_shapes
:@
?
save_2/Assign_9Assigngen_flows/layer_4/conv2d/kernelsave_2/RestoreV2:9*
validate_shape(*
T0*
use_locking(*2
_class(
&$loc:@gen_flows/layer_4/conv2d/kernel*&
_output_shapes
:@@
?
save_2/Assign_10Assigngen_flows/layer_5/conv2d/biassave_2/RestoreV2:10*
validate_shape(*
use_locking(*
T0*0
_class&
$"loc:@gen_flows/layer_5/conv2d/bias*
_output_shapes
:@
?
save_2/Assign_11Assigngen_flows/layer_5/conv2d/kernelsave_2/RestoreV2:11*
use_locking(*2
_class(
&$loc:@gen_flows/layer_5/conv2d/kernel*&
_output_shapes
:@@*
T0*
validate_shape(
?
save_2/Assign_12Assigngen_flows/layer_6/conv2d/biassave_2/RestoreV2:12*0
_class&
$"loc:@gen_flows/layer_6/conv2d/bias*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
?
save_2/Assign_13Assigngen_flows/layer_6/conv2d/kernelsave_2/RestoreV2:13*2
_class(
&$loc:@gen_flows/layer_6/conv2d/kernel*&
_output_shapes
:@ *
validate_shape(*
T0*
use_locking(
?
save_2/Assign_14Assigngen_flows/layer_7/conv2d/biassave_2/RestoreV2:14*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*0
_class&
$"loc:@gen_flows/layer_7/conv2d/bias
?
save_2/Assign_15Assigngen_flows/layer_7/conv2d/kernelsave_2/RestoreV2:15*
T0*
use_locking(*&
_output_shapes
:  *2
_class(
&$loc:@gen_flows/layer_7/conv2d/kernel*
validate_shape(
?
save_2/Assign_16Assign#gen_flows/outputs_flows/conv2d/biassave_2/RestoreV2:16*
validate_shape(*
T0*
use_locking(*6
_class,
*(loc:@gen_flows/outputs_flows/conv2d/bias*
_output_shapes
:
?
save_2/Assign_17Assign%gen_flows/outputs_flows/conv2d/kernelsave_2/RestoreV2:17*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(*8
_class.
,*loc:@gen_flows/outputs_flows/conv2d/kernel
?
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_2^save_2/Assign_3^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard "&B
save_2/Const:0save_2/Identity:0save_2/restore_all (5 @F8"?
	variables??
?
!gen_flows/layer_0/conv2d/kernel:0&gen_flows/layer_0/conv2d/kernel/Assign&gen_flows/layer_0/conv2d/kernel/read:02;gen_flows/layer_0/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_0/conv2d/bias:0$gen_flows/layer_0/conv2d/bias/Assign$gen_flows/layer_0/conv2d/bias/read:021gen_flows/layer_0/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_1/conv2d/kernel:0&gen_flows/layer_1/conv2d/kernel/Assign&gen_flows/layer_1/conv2d/kernel/read:02;gen_flows/layer_1/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_1/conv2d/bias:0$gen_flows/layer_1/conv2d/bias/Assign$gen_flows/layer_1/conv2d/bias/read:021gen_flows/layer_1/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_2/conv2d/kernel:0&gen_flows/layer_2/conv2d/kernel/Assign&gen_flows/layer_2/conv2d/kernel/read:02;gen_flows/layer_2/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_2/conv2d/bias:0$gen_flows/layer_2/conv2d/bias/Assign$gen_flows/layer_2/conv2d/bias/read:021gen_flows/layer_2/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_3/conv2d/kernel:0&gen_flows/layer_3/conv2d/kernel/Assign&gen_flows/layer_3/conv2d/kernel/read:02;gen_flows/layer_3/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_3/conv2d/bias:0$gen_flows/layer_3/conv2d/bias/Assign$gen_flows/layer_3/conv2d/bias/read:021gen_flows/layer_3/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_4/conv2d/kernel:0&gen_flows/layer_4/conv2d/kernel/Assign&gen_flows/layer_4/conv2d/kernel/read:02;gen_flows/layer_4/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_4/conv2d/bias:0$gen_flows/layer_4/conv2d/bias/Assign$gen_flows/layer_4/conv2d/bias/read:021gen_flows/layer_4/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_5/conv2d/kernel:0&gen_flows/layer_5/conv2d/kernel/Assign&gen_flows/layer_5/conv2d/kernel/read:02;gen_flows/layer_5/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_5/conv2d/bias:0$gen_flows/layer_5/conv2d/bias/Assign$gen_flows/layer_5/conv2d/bias/read:021gen_flows/layer_5/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_6/conv2d/kernel:0&gen_flows/layer_6/conv2d/kernel/Assign&gen_flows/layer_6/conv2d/kernel/read:02;gen_flows/layer_6/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_6/conv2d/bias:0$gen_flows/layer_6/conv2d/bias/Assign$gen_flows/layer_6/conv2d/bias/read:021gen_flows/layer_6/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_7/conv2d/kernel:0&gen_flows/layer_7/conv2d/kernel/Assign&gen_flows/layer_7/conv2d/kernel/read:02;gen_flows/layer_7/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_7/conv2d/bias:0$gen_flows/layer_7/conv2d/bias/Assign$gen_flows/layer_7/conv2d/bias/read:021gen_flows/layer_7/conv2d/bias/Initializer/Const:08
?
'gen_flows/outputs_flows/conv2d/kernel:0,gen_flows/outputs_flows/conv2d/kernel/Assign,gen_flows/outputs_flows/conv2d/kernel/read:02Agen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal:08
?
%gen_flows/outputs_flows/conv2d/bias:0*gen_flows/outputs_flows/conv2d/bias/Assign*gen_flows/outputs_flows/conv2d/bias/read:027gen_flows/outputs_flows/conv2d/bias/Initializer/Const:08"?
trainable_variables??
?
!gen_flows/layer_0/conv2d/kernel:0&gen_flows/layer_0/conv2d/kernel/Assign&gen_flows/layer_0/conv2d/kernel/read:02;gen_flows/layer_0/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_0/conv2d/bias:0$gen_flows/layer_0/conv2d/bias/Assign$gen_flows/layer_0/conv2d/bias/read:021gen_flows/layer_0/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_1/conv2d/kernel:0&gen_flows/layer_1/conv2d/kernel/Assign&gen_flows/layer_1/conv2d/kernel/read:02;gen_flows/layer_1/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_1/conv2d/bias:0$gen_flows/layer_1/conv2d/bias/Assign$gen_flows/layer_1/conv2d/bias/read:021gen_flows/layer_1/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_2/conv2d/kernel:0&gen_flows/layer_2/conv2d/kernel/Assign&gen_flows/layer_2/conv2d/kernel/read:02;gen_flows/layer_2/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_2/conv2d/bias:0$gen_flows/layer_2/conv2d/bias/Assign$gen_flows/layer_2/conv2d/bias/read:021gen_flows/layer_2/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_3/conv2d/kernel:0&gen_flows/layer_3/conv2d/kernel/Assign&gen_flows/layer_3/conv2d/kernel/read:02;gen_flows/layer_3/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_3/conv2d/bias:0$gen_flows/layer_3/conv2d/bias/Assign$gen_flows/layer_3/conv2d/bias/read:021gen_flows/layer_3/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_4/conv2d/kernel:0&gen_flows/layer_4/conv2d/kernel/Assign&gen_flows/layer_4/conv2d/kernel/read:02;gen_flows/layer_4/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_4/conv2d/bias:0$gen_flows/layer_4/conv2d/bias/Assign$gen_flows/layer_4/conv2d/bias/read:021gen_flows/layer_4/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_5/conv2d/kernel:0&gen_flows/layer_5/conv2d/kernel/Assign&gen_flows/layer_5/conv2d/kernel/read:02;gen_flows/layer_5/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_5/conv2d/bias:0$gen_flows/layer_5/conv2d/bias/Assign$gen_flows/layer_5/conv2d/bias/read:021gen_flows/layer_5/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_6/conv2d/kernel:0&gen_flows/layer_6/conv2d/kernel/Assign&gen_flows/layer_6/conv2d/kernel/read:02;gen_flows/layer_6/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_6/conv2d/bias:0$gen_flows/layer_6/conv2d/bias/Assign$gen_flows/layer_6/conv2d/bias/read:021gen_flows/layer_6/conv2d/bias/Initializer/Const:08
?
!gen_flows/layer_7/conv2d/kernel:0&gen_flows/layer_7/conv2d/kernel/Assign&gen_flows/layer_7/conv2d/kernel/read:02;gen_flows/layer_7/conv2d/kernel/Initializer/random_normal:08
?
gen_flows/layer_7/conv2d/bias:0$gen_flows/layer_7/conv2d/bias/Assign$gen_flows/layer_7/conv2d/bias/read:021gen_flows/layer_7/conv2d/bias/Initializer/Const:08
?
'gen_flows/outputs_flows/conv2d/kernel:0,gen_flows/outputs_flows/conv2d/kernel/Assign,gen_flows/outputs_flows/conv2d/kernel/read:02Agen_flows/outputs_flows/conv2d/kernel/Initializer/random_normal:08
?
%gen_flows/outputs_flows/conv2d/bias:0*gen_flows/outputs_flows/conv2d/bias/Assign*gen_flows/outputs_flows/conv2d/bias/read:027gen_flows/outputs_flows/conv2d/bias/Initializer/Const:08*?
serving_default?
0
input_N%
Placeholder_1:0
-
inputs#
Placeholder:0
8
Neighbors_img'
Placeholder_2:0??
9
Neighbors_flow'
Placeholder_3:0??)
im_out
Sum_2:0??tensorflow/serving/predict