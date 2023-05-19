# PVN3D

[Paper](https://arxiv.org/pdf/1911.04231.pdf), [Code](https://github.com/ethnhe/PVN3D)

Deep 3D keypoints [Hough voting]() network with instance/semantic segmentation for 6DoF pose estimation on RGB-D image.
![[Pasted image 20221123104726.png]]

---
##  Network Architecture

**Feature-extraction Stage**:
1. CNN (PSPNet + ResNet34)
	- ImageNet-pretrained. Extract RGB appearance information
2. PointNet++
	- Extract geometry information in point clouds and their normal maps
3. [Dense Fusion](./2_densefusion) ([paper](https://arxiv.org/pdf/1901.04780.pdf))
	- Combine CNN and PointNet features.
After this stage, each point $p_i$ has C-dimensional feature $f_i \in \mathbb{R}^C$


---
## 3D Keypoint Detection (shared MLP)

$$L_{\text {multi-task }}=\lambda_1 L_{\text {keypoints }}+\lambda_2 L_{\text {semantic }}+\lambda_3 L_{\text {center }}$$

### 1. $M_K$ : 3D keypoint detection

**Input** to $M_k$. $\{p_i\}_{i=1}^{N}$ : Set of visible ***seed points***. $\{kp_j\}_{j=1}^{M}$: selected ***keypoint***. 
	- $p_i = [x_i;f_i]$ ($x_i\in R^3$ is 3D seed point coordinate, $f_i$ is the point feature from dense fusion)
	- $kp_i=[y_i]$ ($y_j\in R^3$ is 3D keypoint coordinate)
	- $N=12,288$ (points sampled from 1 RGB-D image frame)

Within $M_k$. $\{of_i^j\}_{j=1}^M$: Translation offset from (one) i-th seed point to (all) j-th keypoint.  
	- $vkp_i^j=x_i+of_i^j$: ij-th ***voted*** keypoint.

L1 Loss for leaning translation offset: $L_{\text {keypoints }}=\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^M\left\|o f_i^j-o f_i^{j *}\right\| \mathbb{I}\left(p_i \in I\right)$
	- $of_i^{j *}$: ground-truth translation offset. $\mathbb{I}$: Indicating function (1 only when point $p_i$ belong to instance $I$).

### 2. $M_S$: Point-wise instance segmentation

[Focal Loss](https://paperswithcode.com/method/focal-loss) ([original paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)): $\begin{aligned} L_{\text {semantic }}=&-\alpha\left(1-q_i\right)^\gamma \log \left(q_i\right) \\ & \text { where } q_i=c_i \cdot l_i \end{aligned}$ 
	- $\alpha$: $\alpha$-balance parameter. $\gamma$: Focusing parameter
	- $c_i$: Predicted confidence  for i-th point belonging to each class
	- $l_i$: One-hot ground truth class label

### 3. $M_C$: Center voting (for distinguishing instances)

L1 Loss for center voting: $L_{\text {center }}=\frac{1}{N} \sum_{i=1}^N\left\|\Delta x_i-\Delta x_i^*\right\| \mathbb{I}\left(p_i \in I\right)$ 

#### Keypoint Selection

Keypoints are selected using ***farthest point sampling*** on (ground truth object) mesh.

---
## 6 DoF Pose Estimation

### Least-square fitting

$$
L_{\text {least-squares }}=\sum_{j=1}^M\left\|k p_j-\left(R \cdot k p_j^{\prime}+t\right)\right\|^2
$$

Given 2 point sets.
	- $\{kp_j\}_{j=1}^M$: Detected points in camera frame. 
	- $\{kp_j'\}_{j=1}^M$: Corresponding points in canonical frame.

6D pose estimation module computes $(R, t)$ by minimizing least-square error.
