

namespace e3nn {
    namespace o3 {
        //_angular_spherical_harmonics.py:class SphericalHarmonicsAlphaBeta(torch.nn.Module):
        //_angular_spherical_harmonics.py:class Legendre(fx.GraphModule):
        //experimental/_full_tp.py:class FullTensorProduct(nn.Module):
        //_irreps.py:class Irrep(tuple):
        //_irreps.py:class _MulIr(tuple):
        //_irreps.py:class Irreps(tuple):
        //_linear.py:class Instruction(NamedTuple):
        //_linear.py:class Linear(CodeGenMixin, torch.nn.Module):
        //_norm.py:class Norm(torch.nn.Module):
        //_reduce.py:class ReducedTensorProducts(CodeGenMixin, torch.nn.Module):
        //_s2grid.py:class ToS2Grid(torch.nn.Module):
        //_s2grid.py:class FromS2Grid(torch.nn.Module):
        //_so3grid.py:class SO3Grid(torch.nn.Module):  # pylint: disable=abstract-method
        //_spherical_harmonics.py:class SphericalHarmonics(torch.nn.Module):
        //_tensor_product/_instruction.py:class Instruction(NamedTuple):
        //_tensor_product/_sub.py:class FullyConnectedTensorProduct(TensorProduct):
        //_tensor_product/_sub.py:class ElementwiseTensorProduct(TensorProduct):
        //_tensor_product/_sub.py:class FullTensorProduct(TensorProduct):
        //_tensor_product/_sub.py:class TensorSquare(TensorProduct):
        //_tensor_product/_tensor_product.py:class TensorProduct(CodeGenMixin, torch.nn.Module):
        //_angular_spherical_harmonics.py:def spherical_harmonics_alpha_beta(l, alpha, beta, *, normalization: str = "integral"):
        //_angular_spherical_harmonics.py:def spherical_harmonics_alpha(l: int, alpha: torch.Tensor) -> torch.Tensor:
        //_angular_spherical_harmonics.py:def _poly_legendre(l, m):
        //_angular_spherical_harmonics.py:def _sympy_legendre(l, m) -> float:
        //_angular_spherical_harmonics.py:def _mul_m_lm(mul_l: List[Tuple[int, int]], x_m: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
        //experimental/_full_tp.py:def _prepare_inputs(input1, input2):
        //irrep/__init__.py:def __getattr__(name: str) -> Irrep:
        //_linear.py:def _codegen_linear(
        //_reduce.py:def _wigner_nj(*irrepss, normalization: str = "component", filter_ir_mid=None, dtype=None, device=None):
        //_reduce.py:def _get_ops(path):
        //_rotation.py:def rand_matrix(*shape, requires_grad: bool = False, dtype=None, device=None):
        //_rotation.py:def identity_angles(*shape, requires_grad: bool = False, dtype=None, device=None):
        //_rotation.py:def rand_angles(*shape, requires_grad: bool = False, dtype=None, device=None):
        //_rotation.py:def compose_angles(a1, b1, c1, a2, b2, c2):
        //_rotation.py:def inverse_angles(a, b, c):
        //_rotation.py:def identity_quaternion(*shape, requires_grad: bool = False, dtype=None, device=None):
        //_rotation.py:def rand_quaternion(*shape, requires_grad: bool = False, dtype=None, device=None):
        //_rotation.py:def compose_quaternion(q1, q2) -> torch.Tensor:
        //_rotation.py:def inverse_quaternion(q):
        //_rotation.py:def rand_axis_angle(*shape, requires_grad: bool = False, dtype=None, device=None):
        //_rotation.py:def compose_axis_angle(axis1, angle1, axis2, angle2):
        //_rotation.py:def matrix_x(angle: torch.Tensor) -> torch.Tensor:
        //_rotation.py:def matrix_y(angle: torch.Tensor) -> torch.Tensor:
        //_rotation.py:def matrix_z(angle: torch.Tensor) -> torch.Tensor:
        //_rotation.py:def angles_to_matrix(alpha, beta, gamma) -> torch.Tensor:
        //_rotation.py:def matrix_to_angles(R):
        //_rotation.py:def angles_to_quaternion(alpha, beta, gamma) -> torch.Tensor:
        //_rotation.py:def matrix_to_quaternion(R) -> torch.Tensor:
        //_rotation.py:def axis_angle_to_quaternion(xyz, angle) -> torch.Tensor:
        //_rotation.py:def quaternion_to_axis_angle(q):
        //_rotation.py:def matrix_to_axis_angle(R):
        //_rotation.py:def angles_to_axis_angle(alpha, beta, gamma):
        //_rotation.py:def axis_angle_to_matrix(axis, angle) -> torch.Tensor:
        //_rotation.py:def quaternion_to_matrix(q) -> torch.Tensor:
        //_rotation.py:def quaternion_to_angles(q):
        //_rotation.py:def axis_angle_to_angles(axis, angle):
        //_rotation.py:def angles_to_xyz(alpha, beta) -> torch.Tensor:
        //_rotation.py:def xyz_to_angles(xyz):
        //_s2grid.py:def _quadrature_weights(b, dtype=None, device=None):
        //_s2grid.py:def s2_grid(res_beta, res_alpha, dtype=None, device=None):
        //_s2grid.py:def spherical_harmonics_s2_grid(lmax, res_beta, res_alpha, dtype=None, device=None):
        //_s2grid.py:def _complete_lmax_res(lmax, res_beta, res_alpha):
        //_s2grid.py:def _expand_matrix(ls, like=None, dtype=None, device=None):
        //_s2grid.py:def rfft(x, l) -> torch.Tensor:
        //_s2grid.py:def irfft(x, res):
        //_so3grid.py:def flat_wigner(lmax: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        //_spherical_harmonics.py:def spherical_harmonics(
        //_spherical_harmonics.py:def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        //_spherical_harmonics.py:def _generate_spherical_harmonics(lmax, device=None) -> None:  # pragma: no cover
        //_tensor_product/_codegen.py:def _sum_tensors(xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor) -> torch.Tensor:
        //_tensor_product/_codegen.py:def codegen_tensor_product_left_right(
        //_tensor_product/_codegen.py:def codegen_tensor_product_right(
        //_tensor_product/_sub.py:def _square_instructions_full(irreps_in, filter_ir_out=None, irrep_normalization=None):
        //_tensor_product/_sub.py:def _square_instructions_fully_connected(irreps_in, irreps_out, irrep_normalization=None):
        //_wigner.py:def su2_generators(j: int) -> torch.Tensor:
        //_wigner.py:def change_basis_real_to_complex(l: int, dtype=None, device=None) -> torch.Tensor:
        //_wigner.py:def so3_generators(l) -> torch.Tensor:
        //_wigner.py:def wigner_D(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        //_wigner.py:def wigner_3j(l1: int, l2: int, l3: int, dtype=None, device=None) -> torch.Tensor:
        //_wigner.py:def _so3_clebsch_gordan(l1: int, l2: int, l3: int) -> torch.Tensor:
        //_wigner.py:def _su2_clebsch_gordan(j1: Union[int, float], j2: Union[int, float], j3: Union[int, float]) -> torch.Tensor:
        //_wigner.py:def _su2_clebsch_gordan_coeff(idx1, idx2, idx3):
    }
    namespace nn {
        //_activation.py:class Activation(torch.nn.Module):
        //_batchnorm.py:class BatchNorm(nn.Module):
        //_dropout.py:class Dropout(torch.nn.Module):
        //_extract.py:class Extract(CodeGenMixin, torch.nn.Module):
        //_extract.py:class ExtractIr(Extract):
        //_fc.py:class _Layer(torch.nn.Module):
        //_fc.py:class FullyConnectedNet(torch.nn.Sequential):
        //_gate.py:class _Sortcut(torch.nn.Module):
        //_gate.py:class Gate(torch.nn.Module):
        //_identity.py:class Identity(torch.nn.Module):
        //_normact.py:class NormActivation(torch.nn.Module):
        //_s2act.py:class S2Activation(torch.nn.Module):
        //_so3act.py:class SO3Activation(torch.nn.Module):
    }
    namespace io {
        // Cartesian tensor
        // Spherical tensor
    }
    namespace math {
        // linalg
            // direct sum
            // orthonormalize
            // complete basis
        // normalize_activation
            // moment
            // nomalize2mom
        // perm
        // _reduce
            // germinate_formulas
            // reduce_permutation
        // soft one hot linspace
        // soft unit step
    }
    namespace util {
        // _argtools.py:def _transform(dat, irreps_dat, rot_mat, translation: float = 0.0, output_transform_dtype: bool = False):
        // _argtools.py:def _get_io_irreps(func, irreps_in=None, irreps_out=None):
        // _argtools.py:def _get_args_in(func, args_in=None, irreps_in=None, irreps_out=None):
        // _argtools.py:def _rand_args(irreps_in, batch_size: Optional[int] = None):
        // _argtools.py:def _get_device(mod: torch.nn.Module) -> torch.device:
        // _argtools.py:def _get_floating_dtype(mod: torch.nn.Module) -> torch.dtype:
        // _argtools.py:def _to_device_dtype(args, device=None, dtype=None):
        // default_type.py:def torch_get_default_tensor_type() -> str:
        // default_type.py:def _torch_get_default_dtype() -> torch.dtype:
        // default_type.py:def torch_get_default_device() -> torch.device:
        // default_type.py:def explicit_default_types(dtype: Optional[torch.dtype], device: Optional[torch.device]) -> Tuple[torch.dtype, torch.device]:
    }
}
