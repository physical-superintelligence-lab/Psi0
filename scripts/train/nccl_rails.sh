# Sourced per node from inside the srun/container step to set NCCL_IB_HCA.
#
# Rails are the eight 400Gb InfiniBand ports. They must be selected at runtime,
# per node, for two reasons:
#   * mlx5_* numbering is not uniform: h2-h8 carry ibs6 (the 100Gb spare) and so
#     end at mlx5_10, while h9 carries ibs12 instead and ends at mlx5_9. A single
#     hardcoded list is therefore wrong on h9 -- it drops a real 400Gb rail
#     (mlx5_6) and admits a dead 100Gb port (mlx5_10).
#   * exporting from the batch shell would push the FIRST node's list to every
#     node, which reintroduces the same bug on a heterogeneous allocation.
#
# Autodiscovery (leaving NCCL_IB_HCA unset) is not equivalent: every node also
# exposes a 100Gb IB spare and a 25Gb RoCE bond, both ACTIVE. NCCL has no rate
# filter, so it can bind a GPU to the 100Gb port and run that rail 4x slow.
#
# Selection is by link layer + rate, so it needs no hardcoded device or netdev
# names and no IPoIB addressing. Pure sysfs: works inside the container.

_nccl_hca=""
for _p in /sys/class/infiniband/*/ports/1; do
    [ -r "$_p/link_layer" ] && [ -r "$_p/rate" ] || continue
    [ "$(cat "$_p/link_layer")" = "InfiniBand" ] || continue
    case "$(cat "$_p/rate")" in
        400*) ;;
        *) continue ;;
    esac
    _d="${_p%/ports/1}"
    _nccl_hca="$_nccl_hca,${_d##*/}"
done
_nccl_hca="${_nccl_hca#,}"

if [ -n "$_nccl_hca" ]; then
    export NCCL_IB_HCA="$_nccl_hca"
    echo "NCCL rails on $(hostname): $NCCL_IB_HCA"
else
    echo "WARNING: no 400Gb InfiniBand ports found on $(hostname); leaving NCCL_IB_HCA unset" >&2
fi
unset _nccl_hca _p _d
