

#  Avadhut Chaudhari                                                                                                   Batch 2

##  1. Convolution

It is a mathematical operation which does element wise product and sum of two input signals, with one of the input signal is flipped.

Types of signals convolved:

1. Input signal

2. Kernel / Filter

The purpose of convolution to find, how much two signals are correlated to each other, and extract hidden patterns in those signals.

Applications- 

1. In Audio ( 1 –Dimensional)

2. In Image (2-Dimensional)

Following Images shows convolution process in image.

#### a.  Input image and Filter / Kernel

 ![convolve1](C:\Users\Nwh\Desktop\convolve1.PNG)



#### b.  Result of Convolution

#### ![convolve2](C:\Users\Nwh\Desktop\convolve2.PNG)

#### c.  Feature/ Activation map generation 

#### ![convolve3](C:\Users\Nwh\Desktop\convolve3.PNG)



#### d. Number of filters is same as number of feature/ activation maps

![convolve4](C:\Users\Nwh\Desktop\convolve4.PNG)

## 2. Filters/ Kernels

It is one of the signals in the convolution process, which moves/ slides over on the input image in a given interval, to perform element wise product and sum. There are different types of filters available, can be used according to need. 

Filters used to feature extraction- Blur, Outline, Contrast, Horizontal lines, Vertical lines, Diagonal lines

 2.1 Right Sobel Filter –to extract vertical features

#####     	![kernel1](C:\Users\Nwh\Desktop\kernel1.PNG)

 2.2 Top Sobel Filter –to extract horizontal features

##### 	![kernel2](C:\Users\Nwh\Desktop\kernel2.PNG)

#### a. Shapes with Single Filter

![kernel3](C:\Users\Nwh\Desktop\kernel3.PNG)



#### b. Shapes with 3 Filters and 2D image

![kernel4](C:\Users\Nwh\Desktop\kernel4.PNG)



#### c. Shapes with Single filter and 3-D Image 

![kernel5](C:\Users\Nwh\Desktop\kernel5.PNG)



#### d. Shapes with N filters and 3-D Image 

​      ![kernel6](C:\Users\Nwh\Desktop\kernel6.PNG)

## 3. Epochs

Epoch is a hyper parameter; need to be tuned before training model.  In one epoch all the samples of
training dataset are passed in both forward and backward way only once in neural network. Suppose we have 2000 samples in training dataset and we select batch size = 500, to decide how many samples to be send at one go per iteration. So 4 iterations will complete 1 epoch.

Care has to be taken while choosing epochs value to be optimum. Higher value can lead to over fitting and lower value can lead to under fitting model.     

![epochs](C:\Users\Nwh\Desktop\epochs.PNG)



## 4. 1x1 Convolution

1x1 convolution can be used to overcome the drawback of 3x3 convolution, by not reinterpreting the local data,but combining all channels. It can work with only one pixel at a time and  cannot extract newer features, but can only merge semantically related features from the previous layer.

In 1x1 convolution, 1x1x no of channels of input convolutions. In convolutional nets, 1x1 convolution kernels and full connection table used.​ 1x1 Convolution never be used in the beginning layers to avoid loss of important information or features of image.

![11convolve](C:\Users\Nwh\Desktop\11convolve.PNG)



## 5. 3x3 Convolution

In 3x3 convolution, filter or kernel of size 3x3 used to slide over input image. There are many filters used for blurring, smoothing, and also to find gradient and edges in images, are 3x3 in shape. The 3x3 kernels most widely used than large size kernel, as we can achieve same thing by increasing 3x3 kernel count and also they can be computed faster as compared to larger kernel.

For example, if we have an image of size 5x5 , with 5x5 kernel we get output of 1x1  and with two 3x3 kernel we can get same output 1x1. 

Also for an image of size 7x7, with 7x7 kernel we get output of 1x1 and with three 3x3 kernel we can get same output 1x1 and so on.    ![33convolve](C:\Users\Nwh\Desktop\33convolve.PNG)



## 6. Feature Maps

Main goal is to have small features first and then build complex features on their top. Feature maps helps in this process by providing a separate channel for any feature. Feature maps are are nothing but channels or images.

Feature maps or activation maps are generated as a result of convolution between input image and kernel or filters. The number of feature maps generated always equal to number of filters used in convolution process.     ![convolve4](C:\Users\Nwh\Desktop\convolve4.PNG)

We get different types of features at different feature map stages where convolution applied.

At the beginning we get low level features like edges in the image, after that mid-level features like shapes, blobs and last stage high level features like face or interested objects.

​    ![featuremap2](C:\Users\Nwh\Desktop\featuremap2.PNG)

## 7. Feature Engineering

Most of the times, the given features in the dataset are no sufficient to give satisfactory results. In such case we have to create new features which might help in improving model performance. This can be achieved by transforming training data and augmenting it with additional features. This can be done in feature wise and sample wise data centering and standard normalization, it also might include rotation, horizontal flip, vertical flip, and zoom operation on images.![featureengg](C:\Users\Nwh\Desktop\featureengg.PNG)

## 8. Activation Function

Activation function (f(x)) is used to convert an input signal of a node in neural network to an output signal by helping to decide which information should be filtered or not. This output signal is used as input in next layer in neural network.

Suppose we have a node with inputs x1, x2 and their respective weights w1,w2.

Consider x0 = 1.

So output, y = w0. x0 + w1.  x1 + w2. x2

​    ![activationfun](C:\Users\Nwh\Desktop\activationfun.PNG)

Here activation function gives output in range of –infinity to +infinity. 

But most of measurements are not in negative values. For example age of person cannot be – 20.

So to bring output in realistic values activation functions are used.

Different kinds of activation functions available, few of them enlisted below:

Relu : brings output in range: max(0,input)

Sigmoid : brings output in range: (0,1)

tanh : brings output in range: (-1, 1)

## 9. Receptive Field

The reason behind adding convolution layer to network, to extract features as well as increasing our receptive field. So decision of adding number of convolution layers to network depends upon what receptive field we want to reach. Every time we add a new convolution layer, our receptive field increase by 2x2.  When we building a network care has to be taken to maintain receptive fields.

example: receptive field of network after adding few 3x3 layers

a. two 3x3 layers - receptive field 5x5

b. three 3x3 layers - receptive field 7x7

c. five 3x3 layers - receptive field 11x11

d. seven 3x3 layers - receptive field 15x15 and so on.

In the following image, The local receptive field at yellow channel is  3x3 as it also sees 3x3 pixels. The local receptive field at blue channel is  3x3 as it also sees 3x3 pixels. But the global receptive field at blue is 5x5.![33convolve](C:\Users\Nwh\Desktop\33convolve.PNG)



## 10. How to create an account on GitHub and upload a sample project

#### a. Go to the Github https://github.com/join

![github1](C:\Users\Nwh\Desktop\github1.PNG)

#### b.  Enter a username, valid email address, and password. Use atleast 7 characters, including a number and  a lowercase letter.

#### c. Review the GitHub terms or service and private policy. After creating account you will be diverted to following page

![github2](C:\Users\Nwh\Desktop\github2.PNG)

d. Choose a plan either free or developer with $7 per month

![github3](C:\Users\Nwh\Desktop\github3.PNG)

e. Your Github account is successfully created.



![github4](C:\Users\Nwh\Desktop\github4.PNG)

#### f.  To upload project, click on a new repository tab 

![github5](C:\Users\Nwh\Desktop\github5.PNG)

Enter repository name, description, you can add readme now or later and select create repository tab.

g.  We have to do following steps on command line

![github6](C:\Users\Nwh\Desktop\github6.PNG)

h. Go to command prompt or terminal 

​	consider git already  installed, if not install it

​	1. Go to the path where git installed 

​			![git1](C:\Users\Nwh\Desktop\git1.PNG)

2.  Set Git username and email id	![	git2](C:\Users\Nwh\Desktop\git2.PNG)

3. Go to Project folder and initialize empty Git repository![git3](C:\Users\Nwh\Desktop\git3.PNG)

4. Add all files in the Project folder![git4](C:\Users\Nwh\Desktop\git4.PNG)

5. Commit all files with any name to commit

   ![git5](C:\Users\Nwh\Desktop\git5.PNG)

6. Check the status of commit. It will show nothing, if already committed![git6](C:\Users\Nwh\Desktop\git6.PNG)



7. Add remote origin

   ![git7](C:\Users\Nwh\Desktop\git7.PNG)

8.   Now push existing repository   

   ![git7_5](C:\Users\Nwh\Desktop\git7_5.PNG)

9. Enter the username and password  of Github account



   ![git8](C:\Users\Nwh\Desktop\git8.PNG)

Now your project under version control and public on Github

10. Refresh your Github account page in browser, you can see uploaded project files

    ![git9](C:\Users\Nwh\Desktop\git9.PNG)

11. Click on settings->Github pages -> Source

    ![git9](C:\Users\Nwh\Desktop\git9.PNG)

    ![git10](C:\Users\Nwh\Desktop\git10.PNG)

![git11](C:\Users\Nwh\Desktop\git11.PNG)

Now your Project is live on above marked link.

## 11. 10 examples of use of MathJax in Markdown

##### a. RMSE

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n(y_i -\hat{y_i})^2}
$$

##### b. Logloss

$$
logloss = -\frac{1}{n}\sum_{i=1}^n[y_i log(\hat{y_i}) + (1-y_i) log(1-\hat{y_i})]
$$

##### c. Cosine Similarity
$$
cosinesimilarity = \frac{ \sum_{i =1}^n y_i \hat{y_i}} {\sqrt{ \frac{1}{n} \sum_{i=1}^n y_i^2} \sqrt{\frac{1}{n}\sum_{i=1}^n \hat{y_i}^2}}
$$
##### d. Activation function(f(x)) 

​	2 inputs: x1 and x2 and output: y
$$
y = w_0.x_o + w_1.x_1 + w_2.x_2\\ where,  y\in(-\infty,\infty ) 
$$
##### e. Gradient descent
$$
\theta_i = \theta_i - \eta \frac{\partial J}{\partial \theta_i}
$$

$$
\
$$

##### f. RMSprop

$$
\mu_t = \beta \mu_{t-1} + (1-\beta) (\frac{\partial J}{\partial \theta_{t,i}})^2 \\

\theta_{t,i} = \theta_{t,i} - \frac{\eta}{\sqrt{\mu_t + \in}} \frac{\partial J}{\partial \theta_{t,i}}
$$

##### g. Backpropagation

​	x = input, o = output, wih =  weights between input and hidden layer

​	mu1 = x * wih, who =  weights between hidden and output layer

​        mu2 = h1 * who, h1 = f(mu1) , o = f(mu2)

​	Chain rule:
$$
\frac{\partial E}{\partial w_{ho}} = \frac{\partial E}{\partial o} \times \frac{\partial o}{\partial \mu2} \times \frac{\partial \mu2}{\partial w_{ho}} \\

\frac{\partial E}{\partial w_{ih}} = \frac{\partial E}{\partial o} \times \frac{\partial o}{\partial \mu2} \times \frac{\partial \mu2}{\partial h1}\times \frac{\partial h1}{\partial \mu1}\times \frac{\partial \mu1}{\partial w_{ih}}
$$

##### h.  Matrices

$$
X = \left[\begin{matrix} 1 & 3 & 0 & 0 \\ 1 & 58 & 1 & 1 \\ 0 & 8 & 0 & 0 \\ 0 & 70 & 0 & 1 \\ 1 & 14 & 0 & 0\end{matrix}\right] \\

w_{ih} = \left[\begin{matrix} 0.3 & 0.8 & 0.8 \\ 0.6 & 0.3 & 0.8 \\ 0.8 & 1 & 1 \\ 1 & 1 & 0.2\end{matrix}\right] \\
$$

mu1= X . wih
$$
\mu1_{(5X3)} = X_{(5X4)} . w_{ih (4X3)} \\
\mu1 = \left[\begin{matrix} 2.1 & 1.7 & 3.2\\ 36.9 & 20.2 & 48.4 \\ 4.8 & 2.4 & 6.4 \\ 43 & 22 & 56.2 \\ 8.7 & 5 & 12\end{matrix}\right] \\
$$

##### i.  Integration of f(x) over period a to b

$$
\int_a^b f(x) \mathrm{d}x
$$





##### j.  Fire Neuron or not

$$
y = \begin{cases}
\text {Fire Neuron}, \text {if  X1+ X2+ bias} > 0\\
\text{No fire}, \text {if  X1+ X2+ bias < 0}
\end{cases}
$$


