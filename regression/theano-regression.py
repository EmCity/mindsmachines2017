import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from matplotlib import style 
import theano 
from theano import tensor as T 
import numpy as np

style.use('fivethirtyeight') 
fig = plt.figure() 
ax1 = fig.add_subplot(1,1,1)

trX = np.linspace(-1,1,101) 
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

X = T.scalar() 
Y = T.scalar()

def model (X,w): 
    return X * w

def custom_alpha (y, Y):
    print type(y)
    print type(Y)
    print y - Y
    if 0.0 < (y - Y):
        return 100
    elif 0.0 > (y - Y):
        return 1



w = theano.shared(np.asarray(0., dtype=theano.config.floatX)) 
y = model(X,w)

cost = sum[[(y - Y) / y]^2 * custom_alpha(y, Y)]
#cost = T.mean(T.sqr(y-Y)) 
#cost = sum[[(y - Y) / y]^2 * custom_alpha(y, Y)]
gradient = T.grad(cost=cost, wrt = w) 
updates = [[w,w-gradient * 0.001]]

train = theano.function(inputs=[X,Y], outputs=cost, updates = updates, allow_input_downcast= True)

#for i in range(100): 
   # for x,y in zip (trX,trY): 
    #    train(x,y) 
     #   print (w.eval()) 
        
        
def animate(i): 
    ax1.clear() 
    plt.scatter(trX, trY,  label='Gradient Descent on GPU', 
                    alpha=0.3, edgecolors='none') 
    plt.legend() 
    plt.grid(True) 
    for x,y in zip (trX,trY): 
        train(x,y) 
        #print (w.eval()) 
       
    xs = [-1,1] 
    ys = [-1*w.eval(),w.eval()] 
    ax1.plot(xs,ys) 
    
ani = animation.FuncAnimation(fig, animate, interval = 250)

#plt.show() 