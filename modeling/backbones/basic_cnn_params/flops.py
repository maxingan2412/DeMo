from fvcore.nn.jit_handles import elementwise_flop_counter
# from modeling.fusion_part.mamba import selective_scan_flop_jit

def give_supported_ops():
    return{
            "aten::silu": elementwise_flop_counter(0, 1),
            "aten::gelu": elementwise_flop_counter(0, 1),
            "aten::neg": elementwise_flop_counter(0, 1),
            "aten::exp": elementwise_flop_counter(0, 1),
            "aten::flip": elementwise_flop_counter(0, 1),
            "aten::mul": elementwise_flop_counter(0, 1),
            "aten::div": elementwise_flop_counter(0, 1),
            "aten::softmax": elementwise_flop_counter(0, 2),
            "aten::sigmoid": elementwise_flop_counter(0, 1),
            "aten::add": elementwise_flop_counter(0, 1),
            "aten::add_": elementwise_flop_counter(0, 1),
            "aten::radd": elementwise_flop_counter(0, 1),
            "aten::sub": elementwise_flop_counter(0, 1),
            "aten::sub_": elementwise_flop_counter(0, 1),
            "aten::rsub": elementwise_flop_counter(0, 1),
            "aten::mul_": elementwise_flop_counter(0, 1),
            "aten::rmul": elementwise_flop_counter(0, 1),
            "aten::div_": elementwise_flop_counter(0, 1),
            "aten::rdiv": elementwise_flop_counter(0, 1),
            "aten::cumsum": elementwise_flop_counter(0, 1),
            "aten::ne": elementwise_flop_counter(0, 1),
            "aten::silu_": elementwise_flop_counter(0, 1),
            "aten::dropout_": elementwise_flop_counter(0, 1),
            "aten::log_softmax": elementwise_flop_counter(0, 2),
            "aten::argmax": elementwise_flop_counter(0, 1),
            "aten::one_hot": elementwise_flop_counter(0, 1),
            "aten::flatten": elementwise_flop_counter(0, 0),
            "aten::unflatten": elementwise_flop_counter(0, 0),
            "aten::mean": elementwise_flop_counter(1, 0),
            "aten::sum": elementwise_flop_counter(1, 0),
            "aten::abs": elementwise_flop_counter(0, 1),
            "aten::tanh": elementwise_flop_counter(0, 1),
            "aten::relu": elementwise_flop_counter(0, 1),
            "aten::where": elementwise_flop_counter(0, 1),
            "aten::le": elementwise_flop_counter(0, 1),
            "aten::topk": elementwise_flop_counter(1, 1),
            "aten::sort": elementwise_flop_counter(1, 1),
            "aten::argsort": elementwise_flop_counter(1, 1),
            "aten::scatter": elementwise_flop_counter(1, 1),
            "aten::gather": elementwise_flop_counter(1, 1),
            "aten::adaptive_max_pool2d": elementwise_flop_counter(1, 0),
            "prim::PythonOp.CrossScan": None,  # just some add or reshape transform
            "prim::PythonOp.CrossMerge": None,  # just some add or reshape transform
            # No use in DeMo
            # "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            # "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            # "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            # "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }
