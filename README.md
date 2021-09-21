# Control Theory Final Project: Neural Net Control

## [New library here](https://github.com/Dmarcano/autograd_rs)

This was a small hand crafted library of a neural network for use in my Control Theory Final Project. In the end for my project I ended up creating a small Matlab network rather than use this as properly translating PID closed loop transfer functions back and forth from frequency to time domains was very challenging in pure Rust. 

Then when using this library for general neural network use I found that the overall structure of using the networks was very rigid.
It was then that I found out about automatic differentiation engines and decided to make my own in another library which is linked to [here](https://github.com/Dmarcano/autograd_rs)