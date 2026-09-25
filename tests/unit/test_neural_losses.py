import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('ultralytics')
from robocup_perception.multitask.model import Field, field_loss, detector_kd


def test_unknown_field_has_zero_loss_and_gradient():
    z=torch.randn(1,1,12,12,requires_grad=True)
    loss,_=field_loss(z,torch.zeros_like(z),torch.zeros_like(z))
    loss.backward()
    assert loss.item()==0 and z.grad.abs().sum().item()==0


def test_teacher_extreme_logits_finite():
    student={'scores':torch.tensor([[-80.,0,80.]]*6).reshape(1,6,3).requires_grad_(),
             'boxes':torch.zeros(1,64,3,requires_grad=True)}
    teacher={k:v.detach().clone() for k,v in student.items()}
    loss=detector_kd(student,teacher)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in student.values())


def test_every_field_stage_receives_gradient():
    torch.set_num_threads(1)
    m=Field().eval()
    s4=torch.randn(1,32,160,160,requires_grad=True)
    s8=torch.randn(1,64,80,80,requires_grad=True)
    y=m(s4,s8)
    assert y.shape==(1,1,320,320)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
