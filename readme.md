# MMUIL 
Implementation of the study proposed in the paper <a href="https://dl.acm.org/doi/10.1007/s10115-024-02088-5">MMUIL: Enhancing Multi-Platform User Identity Linkage with Multi-Information"</a>

We have shared all the code related to this research. 

Citing https://github.com/AndyJZhao/NSHE.  

## Main packages
1.0 < torch < 1.8  
3.5 < python <= 3.7.5  

## Usage

```python
1) Feature Embed
python Name_vec.py
python Deepwalk.py
cd ./NSHE/src python train.py

2) fstw
python GAN_MMUIL_fstw.py

2) dblp
python GAN_MMUIL_dblp.py
python Multi-platform_results.py
```

```
Please note that in Table 7 of our article, the correct unit should be seconds (s). We apologize for the mistake in the current version.
```

## Citation
```bibtex
MMUIL: enhancing multi-platform user identity linkage with multi-information
@Article{9590332,
  author={Qian, Zhou and Yihan, Hei and Wei, Chen and Shangfei, Zheng and Lei, Zhao},
  booktitle={KAIS}, 
  title={MMUIL: enhancing multi-platform user identity linkage with multi-information}, 
  year={2024},
  doi={10.1007/s10115-024-02088-5}
}
```


