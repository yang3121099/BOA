import torch
from utils.quant_utils import damping


def get_cholesky_of_inverse(H):
    """Compute upper-triangular U such that U.T @ U = H^{-1}.

    For positive-definite H with Cholesky factor L (H = L @ L.T), the unique
    upper-triangular Cholesky factor of H^{-1} is U = L^{-1}, because:
        U.T @ U = L^{-T} @ L^{-1} = (L @ L.T)^{-1} = H^{-1}

    L^{-1} is computed via torch.linalg.solve_triangular (L @ X = I → X = L^{-1}),
    which is thread-safe and supports batched inputs for GPU parallelism across heads.
    """
    d = H.shape[-1]
    eye = torch.eye(d, device=H.device, dtype=H.dtype).expand_as(H)

    try:
        # Fast path: one batched solve — parallelises across heads on GPU
        L = torch.linalg.cholesky(H)
        return torch.linalg.solve_triangular(L, eye, upper=False)
    except torch.linalg.LinAlgError:
        pass

    # Slow path: per-head with adaptive damping for ill-conditioned heads
    U = torch.zeros_like(H)
    for i in range(len(H)):
        done = False
        while not done:
            try:
                L_i = torch.linalg.cholesky(H[i])
                U[i] = torch.linalg.solve_triangular(L_i, eye[i], upper=False)
                done = True
            except torch.linalg.LinAlgError:
                H[i] = damping(H[i])
    return U


def reorder_col(W, H_col):
    org_shape = W.shape
    hidden_size = org_shape[-1]
    if H_col.shape[0] == 1:  # Common Hessian for all heads
        W = W.view(1, -1, hidden_size)

    perm = torch.argsort(torch.diagonal(H_col, dim1=-2, dim2=-1), dim=-1, descending=True)
    W = torch.gather(W, dim=-1, index=perm.unsqueeze(-2).expand(-1, W.shape[-2], -1))
    H_col = torch.gather(
        torch.gather(H_col, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, H_col.shape[-1])),
        dim=-1, index=perm.unsqueeze(-2).expand(-1, H_col.shape[-2], -1)
    )
    invperm = torch.argsort(perm, dim=-1)
  
    W = W.view(org_shape)

    return W, H_col, invperm


def reverse_reorder_col(W, invperm):
    org_shape = W.shape
    hidden_size = W.shape[-1]

    W = W.reshape(invperm.shape[0], -1, hidden_size)
    W = torch.gather(W, dim=-1, index=invperm.unsqueeze(-2).expand(-1, W.shape[-2], -1))    
    W = W.reshape(org_shape)
    
    return W


def reorder_row(W, H_row, scale, zero):
    perm = torch.argsort(torch.diagonal(H_row, dim1=-2, dim2=-1), dim=-1, descending=True)
    W = torch.gather(W, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, W.shape[-1]))
    H_row = torch.gather(
        torch.gather(H_row, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, H_row.shape[-1])),
        dim=-1, index=perm.unsqueeze(-2).expand(-1, H_row.shape[-2], -1)
    )
    scale = torch.gather(scale, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, scale.shape[-1]))
    zero = torch.gather(zero, dim=-2, index=perm.unsqueeze(-1).expand(-1, -1, zero.shape[-1]))
    invperm = torch.argsort(perm, dim=-1)


    return W, H_row, scale, zero, invperm 
    

def reverse_reorder_row(W, invperm_row):
    W = torch.gather(W, dim=-2, index=invperm_row.unsqueeze(-1).expand(-1, -1, W.shape[-1]))    
        
    return W