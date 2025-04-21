# models/modules/ode.py

"""
Neural ODE Module (ode.py)

본 모듈은 PyTorch 기반 Neural ODE를 구현한 예시입니다.
ODEFunc(nn.Module)을 통해 미분 방정식을 정의하고,
ODEBlock(nn.Module)을 통해 odeint를 이용하여 전방해석(Forward Integration)을 수행합니다.

Usage:
    from models.modules.ode import ODEModel

    # input_dim: 입력 벡터 차원
    # hidden_dim: ODE 함수 내부의 MLP hidden 차원
    model = ODEModel(input_dim=32, hidden_dim=64)

    # forward 수행
    x = torch.randn(batch_size, 32)
    out = model(x)
"""

import torch
import torch.nn as nn

# pip install git+https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """
    ODE 함수 f(t, x)를 정의하는 모듈.
    일반적으로 x(t)에 대해 미분방정식을 세우고, 
    내부에 MLP, CNN, RNN 등 임의의 신경망 구조를 사용 가능.
    """
    def __init__(self, in_features: int, hidden_dim: int):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features)
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        f(t, x) = dx/dt를 계산하여 반환.
        Neural ODE에서는 t를 명시적으로 사용하지 않는 경우가 많지만,
        odeint 함수에 입력으로 t가 필요하여 인자로만 받아둠.
        """
        return self.net(x)


class ODEBlock(nn.Module):
    """
    ODEFunc를 이용하여 odeint를 실행하는 블록.
    torchdiffeq의 odeint 함수를 통해 가변적인 t 구간에 대해 적분할 수 있음.
    """
    def __init__(
        self, 
        odefunc: nn.Module, 
        method: str = 'rk4', 
        rtol: float = 1e-5, 
        atol: float = 1e-5
    ):
        """
        Args:
            odefunc (nn.Module): ODEFunc (또는 미분계수를 정의한 nn.Module)
            method (str): 적분 방법 (e.g. 'rk4', 'dopri5' 등)
            rtol (float): 적분 허용 오차 (상대)
            atol (float): 적분 허용 오차 (절대)
        """
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(
        self,
        x: torch.Tensor, 
        integration_times: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [batch_size, in_features] 형태의 입력
            integration_times (torch.Tensor): e.g. torch.tensor([0, 1]),
                통합(적분)을 수행할 시간 구간.
                시간 구간이 여러 개면 더 복잡한 trajectory가 반환될 수 있음.

        Returns:
            (torch.Tensor): [batch_size, in_features] 형태의 적분 결과(마지막 시점)
        """
        if integration_times is None:
            # 예시로 t=0 ~ t=1까지 적분
            integration_times = torch.tensor([0.0, 1.0]).float().to(x.device)

        # odeint는 [seq_len, batch, features] 형태로 반환
        out = odeint(
            func=self.odefunc,
            y0=x,
            t=integration_times,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol
        )
        # 마지막 시점의 결과만 사용
        return out[-1]


class ODEModel(nn.Module):
    """
    Neural ODE 전체 모델 예시.
    ODEFunc, ODEBlock을 순차적으로 결합하여 forward를 정의.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        method: str = 'rk4'
    ):
        """
        Args:
            input_dim (int): 입력 차원
            hidden_dim (int): 내부 MLP hidden 차원
            method (str): 적분 방법
        """
        super(ODEModel, self).__init__()
        self.odefunc = ODEFunc(input_dim, hidden_dim)
        self.ode_block = ODEBlock(self.odefunc, method=method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ode_block(x)
