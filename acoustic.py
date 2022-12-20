import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from examples.seismic import Model, plot_velocity, TimeAxis, RickerSource,Receiver,plot_shotrecord
from devito import TimeFunction
from devito import Eq, solve
from devito import Operator


class acoustic_model:

    def __init__(self):
        self.set_default()
    
    def set_default(self):
        self.set_two_layer_model()
        self.set_model_time(2000.0)
        self.set_ricker_src(2500,40)
        rN = 101
        rx = np.linspace(0,5000,rN)
        rz = np.ones(rN)*40
        self.set_receiver(rx,rz)

    def set_model(self,model):
        self.model = model
    
    def set_two_layer_model(self,
            x_range = 5000,
            y_range = 2000,
            grid_size = 10,
            boundary_depth=1000,
            v1 = 1.5,
            v2 = 2.5  
        ):
        '''
        x_range, y_range, grid_size, boundary_depth are in meter
        v1 and v2 are in km/s
        '''

        Nx = x_range//grid_size
        Ny = y_range//grid_size
        shape = (Nx,Ny)
        spacing = (grid_size,grid_size)
        origin = (0,0)

        v = np.empty(shape,dtype=np.float32)

        boundary_ind = boundary_depth//grid_size
        v[:,:boundary_ind] = v1
        v[:,boundary_ind:] = v2
        model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                space_order=8, nbl=10, bcs="damp")

        self.model = model
    
    def plot_model(self,src_rec_on=True,rec_downsample = 1):
        if src_rec_on:
            plot_velocity(self.model, 
                source=self.src.coordinates.data,
                receiver=self.rec.coordinates.data[::rec_downsample, :])
        else:
            plot_velocity(self.model)

    def set_model_time(self,total_time):
        '''
        total_time: total model simulation time in ms
        '''
        t0 = 0
        tn = total_time
        dt = self.model.critical_dt

        self.time_range = TimeAxis(start=t0,stop=tn,step=dt)
        self.total_time = total_time
        self.dt = dt
    
    def set_ricker_src(self,sx,sz,f0=10):
        src = RickerSource(name='src', grid=self.model.grid, 
                    f0=f0/1000.0,
                    npoint=1, time_range=self.time_range)        
        
        src.coordinates.data[0, :] = sx
        src.coordinates.data[0, -1] = sz 
        self.src = src
    
    def set_src(self,src):
        self.src = src

    def plot_src_wavelet(self):
        self.src.show()
    
    def set_receiver(self,rx,rz):
        '''
        rx and rz should be 1D numpy array in same length
        unit in meters
        '''
        rec = Receiver(name='rec',
                grid=self.model.grid,
                npoint=len(rx),
                time_range = self.time_range
                )
        
        rec.coordinates.data[:,0] = rx
        rec.coordinates.data[:,1] = rz

        self.rec = rec

    def run_model(self):
        model = self.model
        src = self.src
        rec = self.rec
        time_range = self.time_range
        dt = self.dt

        # Define the wavefield with the size of the model and the time dimension
        u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2, save=time_range.num)

        # We can now write the PDE
        pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

        stencil = Eq(u.forward, solve(pde, u.forward))

        # Finally we define the source injection and receiver read function to generate the corresponding code
        src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)

        # Create interpolation expression for receivers
        rec_term = rec.interpolate(expr=u.forward)

        u.data[:]=0
        op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)

        results = op(dt=model.critical_dt)

        self.pde = pde
        self.results = results
        self.u = u

    def plot_shotrecord(self):
        plot_shotrecord(self.rec.data,self.model,0,self.total_time)
    
    def plot_wavefield_bytime(self,time=0):
        """
        Time should be in ms
        """
        xmin,ymin = self.model.grid.origin
        xrange,yrange = self.model.grid.extent
        xmax = xmin+xrange
        ymax = ymin+yrange
        extent = [xmin, xmax, ymax, ymin]
        time_index = int(time/self.dt)
        imax = plt.imshow(self.u.data[time_index].T,extent=extent,cmap='gray')
        return imax
        
    def make_wavefield_movie(self,filename,timestep_skip=20):
        mod = self
        data = mod.u.data

        ## . . Animate Solution and compute error
        tskip = timestep_skip
        frameN = data.shape[0]//tskip - 1
        amax = np.max(np.abs(data))

        ## . . Set up movie
        fig = plt.figure()
        im_ax = mod.plot_wavefield_bytime(0)
        plt.clim(np.array([-1,1])*amax*0.1)

        def AWE_2D_animate(i):
            x = data[i*tskip,:,:]
            im_ax.set_data(x.T)
            t = mod.dt*i*tskip    
            plt.title('Time: {:.1f} ms'.format(t))

        ## . . Call the animator
        anim = animation.FuncAnimation(fig,AWE_2D_animate,frames=frameN,interval=100)
        anim.save(filename)
        plt.close()

    def HTML_display(self,filename,width=500):
        return HTML(f"""
        <img src="{filename}" width="{width}" />
        """)