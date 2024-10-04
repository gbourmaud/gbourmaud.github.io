import numpy as np

def computeHomographyRectification(K, R21, T21):
    R1g = np.eye(3)
    T1g = np.zeros(3)
    R2g = R21
    T2g = T21.ravel()
    
    Rn1, Rn2 = rectify(R1g, T1g, R2g, T2g)
    
    Hn1 = K @ Rn1 @ np.linalg.inv(K)
    Hn2 = K @ Rn2 @ np.linalg.inv(K)
    
    return Hn1, Hn2, Rn1, Rn2

def rectify(R1g, T1g, R2g, T2g):
    # Centers of projection (unchanged)
    c1 = -T1g @ R1g
    c2 = -T2g @ R2g

    # New x axis (direction of the baseline)
    v1 = c1 - c2
    
    # New y axes (orthogonal to new x and old z)
    v2 = np.cross(R1g[2, :], v1)
    
    # New z axes (orthogonal to baseline and y)
    v3 = np.cross(v1, v2)

    # New extrinsic parameters
    Rng = np.vstack((v1 / np.linalg.norm(v1),
                     v2 / np.linalg.norm(v2),
                     v3 / np.linalg.norm(v3)))
    
    # Rectifying image transformation
    R1 = Rng @ (R1g.T)
    R2 = Rng @ (R2g.T)
    
    return R1, R2