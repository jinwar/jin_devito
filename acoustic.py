import numpy as np

from examples.seismic import Model, plot_velocity, TimeAxis, RickerSource,Receiver,plot_shotrecord
from devito import TimeFunction
from devito import Eq, solve
from devito import Operator

class acoustic_model:

    def __init__(self):
        self.set_default()
    
    def set_default(self):
        self.set_two_layer_model()
        self.set_ricker_src(10,2500,40)
        rN = 101
        rx = np.linspace(0,5000,rN)
        rz = np.ones(rN)*40
        self.set_receiver(rx,rz)

    def set_model(self,model):
        self.model = model
    
    def set_two_layer_model(self,
            x_range = 5000,
            y_range = 5000,
            grid_size = 10,
            boundary_depth=2500,
            v1 = 1.5,
            v2 = 2.5  
        ):
        '''
        x_range, y_range, grid_size, boundary_depth are in meter
        v1 and v2 are in km/s
        '''

        Nx = x_range//spacing
        Ny = y_range//spacing
        shape = (Nx,Ny)
        spacing = (grid_size,grid_size)

        v = np.empty(shape,dtype=np.float32)

        boundary_ind = boundary_depth//grid_size
        v[:,:boundary_ind] = v1
        v[:,boundary_ind:] = v2
        model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                space_order=2, nbl=10, bcs="damp")

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
    
    def set_ricker_src(self,f0,sx,sz):
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
                grid=self.model.grid
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