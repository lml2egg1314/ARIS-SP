# Robust-adaptive-steganography-based-on-dither-modulation-and-modification-with-re-compression
This code is implementation of paper "Robust adaptive steganography based on dither modulation and modification with re-compression"


## Abstract
Traditional adaptive steganography is a technique used for covert communication with high security, but it is invalid in the case of stego images are sent to legal receivers over networks which is lossy, such as JPEG compression of channels. To deal with such problem, robust adaptive steganography is proposed to enable the receiver to extract secret messages from the damaged stego images. Previous works utilize reverse engineering and compression-resistant domain constructing to implement robust adaptive steganography. In this paper, we adopt modification with re-compression scheme to improve the robustness of stego sequences in stego images. To balance security and robustness, we move the embedding domain to the low frequency region of DCT (Discrete Cosine Transform) coefficients to improve the security of robust adaptive steganography. In addition, we add additional check codes to further reduce the average extraction error rate based on the framework of E-DMAS (Enhancing Dither Modulation based robust Adaptive Steganography). Compared with GMAS (Generalized dither Modulation based robust Adaptive Steganography) and E-DMAS, experiment results show that our scheme can achieve strong robustness and improve the security of robust adaptive steganography greatly when the channel quality factor is known.

## 摘要
传统的自适应隐写是一种用于具有高安全性要求的隐蔽通信的技术，但是在将载密图像通过有损网络（例如通道的JPEG压缩）发送给合法接收者的情况下，这是无效的。为了解决这种问题，鲁棒的自适应隐写被提出以使接收方能够从受损的载密图像中提取出秘密消息。先前的工作利用逆向工程和抗压缩域构造来实现鲁棒自适应隐写。在本文中，我们采用压缩修改方案提升载密图像中载密序列的鲁棒性。为了平衡安全性和鲁棒性，我们将嵌入域移到DCT（离散余弦变换）系数块的低频区域，以提高鲁棒自适应隐写的安全性。此外，我们基于E-DMAS（基于增强抖动调制的鲁棒自适应隐写）的框架，添加了额外的校验码以进一步降低秘密信息的平均提取错误率。与GMAS（基于广义抖动调制的鲁棒自适应隐写术）和E-DMAS相比，实验结果表明，在已知信道质量因子的情况下，该方案可以实现强大的鲁棒性，并大大提高了鲁棒自适应隐写的安全性。


