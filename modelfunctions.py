import numpy as np
from IPython import display  # used for clearing step numbers, to view during the simulation

class simmodel:
    def __init__(self,numparticles,numsteps,savestep=15):
        self.numparticles = numparticles
        self.eta = 1 # scaling to decrease turning amplitude with speed.  eta=1 means no change
        self.mu_s = np.ones(numparticles)

        self.tau_turn = np.ones(numparticles)*0.1
        self.tau_speed = np.ones(numparticles)
        self.sigma_speed = 0.6 
        self.sigma_turn = 0.6 # 
        self.socialweight = 0.5  # 2 is approx the same as Ioaunnou model, but its too high for this - they always stay together then
        self.mean_mu_s = 1  # should be 1.  This is used for setting the size of the repulsion, align, attract zones
        self.r_repulsion = 3.657*self.mean_mu_s
        self.r_align = 6.857*self.mean_mu_s
        self.r_attract = 20*self.mean_mu_s
        self.ignoresteps=[200,10**4]  # [how many steps to ignore social, how often to do it]
        self.maxturnangle = 10*(np.pi/180)
        self.viewingzone=np.pi

        # multiplication factors for weighting, when summing over particles.
        self.socialmult = np.ones((numparticles,numparticles)) 
        
        # speed social params
        self.speedsocialweight = 0  # "gamma" this defaults to zero
        self.speedzoneweights = [0.5,1] # multipliers for repulsion zone, attraction zone
        self.speed_decelaccelweights = [1,1] # for incorporating asymmetric abilities to slow down vs speed up
        
        # use this to have an agent simply go straight the whole time
        self.strictdirection = False 

        # Stop-go parameters
        self.stopgosim=False
        self.numstates=2  # [go, stop]
        self.Tswitch = 10*np.ones((numparticles,self.numstates))
        self.statespeedmult = np.ones((numparticles,self.numstates))
        self.statespeedmult[:,1] = 0.2
        self.stopgosocial = 0.8
        self.sigma_stopgo=0.1        

        # Configuration
        self.numsimsteps=numsteps  #really, should do at least 10**6, probably more, to ensure sampling enough
        self.savestep=savestep  # don't save all the simulation results - only save this many timesteps
        self.dt = 1/10
        self.xsize, self.ysize = [60,60]
        self.numsavesteps=np.floor(self.numsimsteps/self.savestep).astype(int)    
     
        
    def getdiffs(self,ptcls):
        return diffs_periodic(ptcls,self.xsize,self.ysize)

    
    # Model function, to get zonal desired direction
    def getModelDirections(self,diff,angles):
        # just use the optimized function for all
    
        # keep these as single numbers, or add a dimension if they are a single array. (make for proper element-wise comparison)
        float_or_int = lambda x:  (type(x)==float)|(type(x)==int)
        r_r = self.r_repulsion if float_or_int(self.r_repulsion) else np.array(self.r_repulsion)[:,np.newaxis]
        r_a = self.r_attract if float_or_int(self.r_attract) else np.array(self.r_attract)[:,np.newaxis]
        r_o = self.r_align if float_or_int(self.r_align) else np.array(self.r_align)[:,np.newaxis]
        
        mixsocialzone = np.all(r_a == r_o)

        # Initialize arrays
        repdir = np.zeros((self.numparticles, 2))
        aligndir = np.zeros((self.numparticles, 2))
        attractdir = np.zeros((self.numparticles, 2))
        repspeed = np.zeros(self.numparticles)
        socialspeed = np.zeros(self.numparticles)
        numrep = np.zeros(self.numparticles, dtype=int)

        # Calculate rotated coordinates and other required quantities for all particles
        xrot = np.cos(-angles)
        yrot = np.sin(-angles)
        rotation_matrices = np.array([[xrot, -yrot], [yrot, xrot]]).swapaxes(2,0).swapaxes(1,2)
        coords_rotated = np.einsum('ijk,ilk->ijl', diff[:, :, 0:2], rotation_matrices)
        distances = diff[:, :, 2]
        np.fill_diagonal(distances, np.inf)
        viewing_angles = angles[:, np.newaxis] - np.arctan2(diff[:, :, 1], diff[:, :, 0])
        in_viewing_zone = (np.cos(viewing_angles) > np.cos(self.viewingzone)) & (distances > 0)

        # Zones
        in_repulsion_zone = (distances <= r_r) & in_viewing_zone
        in_orientation_zone = (distances <= r_o) & in_viewing_zone
        in_attraction_zone = (distances <= r_a) & in_viewing_zone
        if not mixsocialzone:
            in_attraction_zone &= (distances > r_o)

        # Repulsion
        repulsion_contrib = -diff[:, :, 0:2] / distances[:, :, np.newaxis]
        repdir = np.sum(repulsion_contrib * in_repulsion_zone[:, :, np.newaxis], axis=1)
        numrep = np.sum(in_repulsion_zone, axis=1)
        repspeed = np.sum(-coords_rotated[:, :, 0] * in_repulsion_zone, axis=1)

        # Alignment
        vlen_diff = np.linalg.norm(diff[:, :, 3:5], axis=2)
        vlen_diff[vlen_diff == 0] = np.inf
        valid_alignment = (vlen_diff > 0) & in_orientation_zone
        alignment_contrib = self.socialmult[:, :, np.newaxis] * diff[:, :, 3:5] / vlen_diff[:, :, np.newaxis]
        aligndir = np.sum(alignment_contrib * valid_alignment[:, :, np.newaxis], axis=1)

        # Attraction
        attraction_contrib = self.socialmult[:, :, np.newaxis] * diff[:, :, 0:2] / distances[:, :, np.newaxis]
        attractdir = np.sum(attraction_contrib * in_attraction_zone[:, :, np.newaxis], axis=1)

        # Speed adjustment in social zones
        dxrot = np.abs(coords_rotated[:, :, 0])
        if mixsocialzone:
            social_zone = (dxrot > r_r) & (dxrot <= r_a) & in_attraction_zone
        else:
            social_zone = (dxrot <= r_a) & (dxrot > r_o) & in_attraction_zone
        socialspeed = np.sum(coords_rotated[:, :, 0] * social_zone, axis=1)

        # Determine new direction and speed for each particle
        newdir = np.zeros((self.numparticles, 2))
        dspeed = np.zeros(self.numparticles)
        for i in range(self.numparticles):
            if numrep[i] > 0:
                newdir[i] = repdir[i]
                mult = self.speed_decelaccelweights[int(repspeed[i] > 0)]
                dspeed[i] = self.speedzoneweights[0] * np.sign(repspeed[i]) * mult
            else:
                newdir[i] = aligndir[i] + attractdir[i]
                mult = self.speed_decelaccelweights[int(socialspeed[i] > 0)]
                dspeed[i] = self.speedzoneweights[1] * np.sign(socialspeed[i]) * mult

            vlennewdir = np.linalg.norm(newdir[i])
            if vlennewdir > 0:
                newdir[i] /= vlennewdir

        return newdir, dspeed   


########################################################################################################################################
#### Distance functions
########################################################################################################################################


def correctdiff_vectorized(diffs, xsize, ysize):
    # Vectorized correction for periodic boundaries
    diffs[:, 0] = np.where(diffs[:, 0] < -xsize / 2, diffs[:, 0] + xsize, diffs[:, 0])
    diffs[:, 0] = np.where(diffs[:, 0] > xsize / 2, diffs[:, 0] - xsize, diffs[:, 0])
    diffs[:, 1] = np.where(diffs[:, 1] < -ysize / 2, diffs[:, 1] + ysize, diffs[:, 1])
    diffs[:, 1] = np.where(diffs[:, 1] > ysize / 2, diffs[:, 1] - ysize, diffs[:, 1])
    return diffs


def diffs_periodic(ptcls, xsize, ysize):
    numparticles = len(ptcls)
    pos = ptcls[:, 0:2]
    vel = ptcls[:, 2:4]

    # Initialize the result array
    result = np.empty((numparticles, numparticles, 5))

    # Calculate position differences and apply periodic boundary conditions
    diffs = pos - pos[:, np.newaxis]
    diffs = correctdiff_vectorized(diffs.reshape(-1, 2), xsize, ysize).reshape(numparticles, numparticles, 2)

    # Calculate distances
    dists = np.sqrt(np.sum(diffs**2, axis=2))

    # Calculate velocity differences
    vdiffs = vel - vel[:, np.newaxis]

    # Combine the results
    result = np.concatenate((diffs, dists[:, :, np.newaxis], vdiffs), axis=2)

    return result





########################################################################################################################################
# Simulation function with loop
########################################################################################################################################

minspeed=0 # minimum speed, because a speed of zero gives errors for dividing by vector length

def ptwsimulation(simmodel, showprogress=False,):
    numparticles, eta, mu_s, tau_turn, tau_speed, sigma_speed, sigma_turn, socialweight = simmodel.numparticles, simmodel.eta, simmodel.mu_s, simmodel.tau_turn, simmodel.tau_speed, simmodel.sigma_speed, simmodel.sigma_turn, simmodel.socialweight
    mean_mu_s = simmodel.mean_mu_s
    
    numstates, stopgosim, statespeedmult, Tswitch, stopgosocial, sigma_stopgo = simmodel.numstates, simmodel.stopgosim, simmodel.statespeedmult, simmodel.Tswitch, simmodel.stopgosocial, simmodel.sigma_stopgo

    numsimsteps, savestep, dt, xsize, ysize, numsavesteps = simmodel.numsimsteps, simmodel.savestep, simmodel.dt, simmodel.xsize, simmodel.ysize, simmodel.numsavesteps
    speedsocialweight = simmodel.speedsocialweight
    
    allparticles = np.zeros([numsavesteps, numparticles, 9])

    [startxpositions,startypositions]=[xsize*np.random.rand(numparticles),ysize*np.random.rand(numparticles)];
    startangles=2*np.pi*np.random.rand(numparticles)-np.pi;
    startspeeds=mu_s*np.ones([numparticles])

    startparticles = np.transpose([startxpositions, # x
                                   startypositions, # y
                                   startspeeds*np.cos(startangles), # vx
                                   startspeeds*np.sin(startangles), # vy
                                   startspeeds, # speed
                                   startangles, # orientation
                                   np.zeros([numparticles]), # angular velocity
                                   np.random.rand(numparticles), # stop-go accumulator variable
                                   np.random.randint(numstates,size=numparticles)  # 'state'
                                   ])
    allparticles[0]=startparticles

    # for optimizing running times and making code easier:
    ind_x, ind_y, ind_vx, ind_vy, ind_spd, ind_ang, ind_angvel, ind_stopgo, ind_state = np.arange(9)
    lnp = np.arange(numparticles)

    currentparticles=startparticles

    # this makes it so that the last entry of "allparticles" won't be zeros, if savestep evenly divides numsimsteps
    if savestep==1:
        numrun = numsimsteps
    else:
        numrun = numsimsteps+1

    step = 1
    ss = 0

    while ss<numsavesteps:
        
        # social interactions via zonal model
        diff=simmodel.getdiffs(currentparticles) 
        cmdirs, cmspeed = simmodel.getModelDirections(diff,currentparticles[:,ind_ang])

        # ignore social interactions if within the range to ignore:
        
        if len(simmodel.ignoresteps)==numparticles:
            ignoremult = np.logical_not((step>simmodel.ignoresteps[:,0]) & (np.mod(step,simmodel.ignoresteps[:,1])-simmodel.ignoresteps[:,0]<0)).astype(int)
        else:
            ignoremult = np.logical_not((step>simmodel.ignoresteps[0]) & (np.mod(step,simmodel.ignoresteps[1])-simmodel.ignoresteps[0]<0)).astype(int)
            ignoremult = np.tile(ignoremult,numparticles)
        cmdirs = cmdirs*ignoremult[:,np.newaxis]
        cmspeed = cmspeed*ignoremult

        vxhat, vyhat = [np.cos(currentparticles[:,ind_ang]), np.sin(currentparticles[:,ind_ang])]  

        socialtorque = socialweight*(vxhat*cmdirs[:,1]-vyhat*cmdirs[:,0])
        # scale torque with speed
        etavals=eta_fn(currentparticles[:,ind_spd]/1,eta)  # leave mu_s here, i.e. don't make it zero
        # updates for angular velocity, orientation, speed
        angvel = currentparticles[:,ind_angvel] + dt/tau_turn*(etavals*socialtorque - currentparticles[:,ind_angvel]) + sigma_turn/np.sqrt(tau_turn)*np.sqrt(dt)*etavals*np.random.randn(numparticles)

        angle = normAngle(currentparticles[:,ind_ang] + dt*angvel)

        if simmodel.strictdirection==True:
            angle[0]=0
            angvel[0]=0
        
        # stop-go state matches
        if stopgosim:
            currentstates = (currentparticles[:,ind_state]).astype(int)
            statematches = np.array([s==currentstates for s in currentstates])*2-1  # +1 for match, -1 for not match
            insocialzone = diff[:,:,2]<simmodel.r_attract
            stateslope = np.array([(np.sum(st[z])-1)/np.sum(z) for st, z in zip(statematches,insocialzone)])
    #         stateslope = (np.sum(statematches,axis=1)-1)/(numparticles-1)  # mean after substract the state of self.  This has the mean going over all space, not just the social zone though
            stopgoaccum = currentparticles[:,ind_stopgo] + dt/Tswitch[lnp,currentstates]*(1-stopgosocial*stateslope) + sigma_stopgo*np.sqrt(dt)*np.random.randn(numparticles)
            stopgoaccum = np.maximum(stopgoaccum,0)
            # see if ones have crossed the threshold.  for these, reset them and switch the states
            toswitch = stopgoaccum>=1
            stopgoaccum[toswitch]=0
            states=currentstates.copy()   # new states
            states[toswitch] = np.mod(states[toswitch]+1,numstates)  # this 'cycles though' states, and could allow for more than 2 states
        else:
            currentstates = np.zeros(numparticles).astype(int)
            states = np.zeros(numparticles).astype(int)
            stopgoaccum = np.zeros(numparticles)

        speed = (currentparticles[:,ind_spd] + dt/tau_speed*(mu_s*statespeedmult[lnp,currentstates] + mu_s*speedsocialweight*cmspeed - currentparticles[:,ind_spd]) 
                 + statespeedmult[lnp,currentstates]*sigma_speed*np.sqrt(dt)*np.random.randn(numparticles)/np.sqrt(tau_speed))
        speed = np.maximum(speed,minspeed)

        # update the velocity
        vx = speed*np.cos(angle)
        vy = speed*np.sin(angle)
        xpos=np.mod(currentparticles[:,ind_x] + dt*vx,xsize)
        ypos=np.mod(currentparticles[:,ind_y] + dt*vy,ysize)
        newparticles = np.stack((xpos,ypos,vx,vy,speed,angle,angvel,stopgoaccum,states),axis=1)    

        # save for this step, if a savestep
        if np.mod(step,savestep)==0:
            allparticles[ss] = newparticles     

        currentparticles=newparticles        

        if np.mod(step,2000)==0:
            if showprogress:
                display.clear_output(wait=True)  
                print(step,numsimsteps)

        step = step+1
        ss=np.floor_divide(step,savestep)

    return allparticles



## calculate quantities for analysis
def dist_and_dcoords(allparticles,simmodel,showprogress=False):
    numsavesteps, numparticles, _ = allparticles.shape
    alldist = np.zeros([numsavesteps,numparticles,numparticles])
    alldcoords = np.zeros([numsavesteps,numparticles,numparticles,2])
    alldcoords_rotated = np.zeros([numsavesteps,numparticles,numparticles,2])
    ind_x, ind_y, ind_vx, ind_vy, ind_spd, ind_ang, ind_angvel = np.arange(7)
    def getmin(vals):
        return np.min(np.reshape(vals,(numparticles,numparticles,-1)),axis=2)
    for ss in range(numsavesteps):
        if np.mod(ss,2000)==0:
            if showprogress:
                display.clear_output(wait=True) 
                print(ss,numsavesteps)
        diff = simmodel.getdiffs(allparticles[ss])    
        dists=np.reshape(diff[:,:,2],(numparticles,numparticles,-1))
        alldist[ss] = np.min(dists,axis=2) 
        am = np.argmin(dists,axis=2) 
            
        # optimized distance calculation (chatGPT)

        # Precompute cosines and sines for all particles
        cosines = np.cos(-allparticles[ss, :, ind_ang])
        sines = np.sin(-allparticles[ss, :, ind_ang])

        # Reshape dx and dy
        dx = np.reshape(diff[:, :, 0], (numparticles, numparticles, -1))
        dy = np.reshape(diff[:, :, 1], (numparticles, numparticles, -1))

        # Use advanced indexing to eliminate the inner loop
        dx_selected = dx[np.arange(numparticles)[:, None], np.arange(numparticles), am]
        dy_selected = dy[np.arange(numparticles)[:, None], np.arange(numparticles), am]

        # Combine dx and dy into a single array
        dcoords = np.stack((dx_selected, dy_selected), axis=-1)

        # Apply rotation
        rotation_matrices = np.array([[cosines, -sines], [sines, cosines]]).transpose(2, 0, 1)
        alldcoords_rotated[ss] = np.einsum('ijk,ikl->ijl', dcoords, rotation_matrices)
            
    return alldist, alldcoords_rotated

def eta_fn(spd_mu, eta,a=1):
    if eta==1:
        return np.ones(len(spd_mu))
    else:
        return a*np.power(eta,1-np.abs(spd_mu))

def normAngle(angles):
    return np.arctan2(np.sin(angles),np.cos(angles))






