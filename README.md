<!-- 
	***
	*   README.md
	*	
	*	Author: Jeong Hoon (Sian) Choi
	*	License: MIT
	*	
	***
-->
<a name="readme-top"></a>

<br/>
<div align="center">
	<a href="https://github.com/csian98/Anomaly-Detection">
		<img src="images/logo.png" alt="Logo" width="150" height="150">
	</a>
	<h3 align="center">DSCI-565: Network Anomaly Detection</h3>	
	<p align="center">
		University of Southern California<br/>
		Viterbi School of Engineering
	<br/>
	<br/>
	<a href="https://github.com/Anomaly-Detection">
	<strong>Explore the docs »</strong>
	</a>
	<br/>
	<br/>
	<a href="https://github.com/Anomaly-Detection/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
	·
	<a href="https://github.com/Anomaly-Detection/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
	</p>
</div>

## About The Project
This project aims tto develop Machine Learning/Deep Learning models for network anomaly detection using the NetFlow datasets.

Additionally, it aims to improve the prediction of attack types that were difficult to predict with existing models by utilizing NetFlow sequence data

The developed models are as follows:

- BaselineFCN
- CNN+LSTM
- NetFlowBERT

Used datasets are as follows:

- NF-UNSW-NB15-v2
- NF-CIC-IDS2018-v2

It was confirmed that using LSTM allowed for the extraction of characteristics of NetFlow data in classification, and that using the Self-Attention layer of BERT more effectively extracted the characteristics of NetFlow data.
Ultimately, using the BERT model, we achieved accuracies of 99.23% and 99.51% on each dataset, and obtained Macro F1 scores of 0.78, 0.73.

### Development Environment
* [![Arch Linux][archlinux-shield]][archlinux-url]
* [![Nvidia Cuda][cuda-shield]][cuda-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## License

Copyright © 2025, *Jeong Hoon Choi* or *Sian*. All rights reserved
Distributed under the MIT License. See `LICENSE` for more information.

## References
- [1] M. Sarhan, S. Layeghy, N. Moustafa, and M. Portmann, ‘NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems’, in Big Data Technologies and Applications, Springer International Publishing, 2021, pp. 117–135.
- [2] L. D. Manocchio, S. Layeghy, W. W. Lo, G. K. Kulatilleke, M. Sarhan, and M. Portmann, ‘FlowTransformer: A transformer framework for flow-based network intrusion detection systems’, Expert Systems with Applications, vol. 241, p. 122564, May 2024.
- [3] P. Sun, P. Liu, Q. Li, C. Liu, X. Lu, R. Hao, and J. Chen, 'DL-IDS: Extracting features using CNN-LSTM hybrid network for intrusion detection system,' Security and Communication Networks, vol 2020, Article ID 8890306, 2020.
- [4] M. S. Sakib and N. Tabassum, 'Analyzing Deep Learning Model Performance for Intrusion Detection on CIC-IDS2017 Dataset,' SSRN Electronic Journal, April 2025.
- [5] M. Sarhan, S. Layeghy, N. Moustafa, and M. Portmann, "NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems," in Big Data Technologies and Applications, Springer, 2021, pp. 117-135.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[macos-shield]: https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0
[macos-url]: https://developer.apple.com/macos
[archlinux-shield]: https://img.shields.io/badge/Arch%20Linux-1793D1?logo=arch-linux&logoColor=fff&style=for-the-badge
[archlinux-url]: https://archlinux.org
[cuda-shield]: https://img.shields.io/badge/NVIDIA%20CUDA-RTX4060-76B900?style=for-the-badge&logo=nvidia&logoColor=white
[cuda-url]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
