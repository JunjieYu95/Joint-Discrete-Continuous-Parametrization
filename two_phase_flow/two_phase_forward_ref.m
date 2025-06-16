%% this matlab function is designed to be called in python via matlab.engine
% it receives a normalized square permeability map and returns the
% mismatch andhe resulting state, this function is used for creating
% reference case used as input for two_phase_forward_with_adjoint

function [obs] = two_phase_forward_ref(Norm_perm,MRST_folder)
% the Norm_perm is normalized log(permeability) range in [0,1]
%% standard setup
% clear all

% always make sure the Norm_perm has the format (N,1)
Norm_perm = reshape(Norm_perm,[],1);

MainDir = cd;
cd(MRST_folder);NewPath = cd;cd(MainDir);
addpath(genpath(NewPath));
startup

%% grid size
nx = int64(sqrt(numel(Norm_perm))); ny = nx; nz = 1;
grid_size = [nx,ny];
Dx = 1000; Dy = 1000; Dz = 10;
dx =  Dx/nx ; dy = Dy/ny; dz = Dz/nz ;
G = cartGrid([nx, ny, nz],[Dx ,Dy ,Dz]);
G = computeGeometry(G, 'Verbose', true);

%% rock property: constant porosity
rock.poro = repmat(0.36 , [G.cells.num, 1]);

%% permeability
% 0 -> 10mD -> log(10) (log(mD))
% 1 -> 200mD -> log(200) log(mD))
min_perm = 50*milli*darcy;
max_perm = 500*milli*darcy;
permx = exp(Norm_perm*(log(max_perm) - log(min_perm))+ log(min_perm));
permy = permx;
permz = 0.1*permx;
rock.perm = [permx permy permz]; %!  constant for test purpose

%% fluid
fluid = initSimpleADIFluid('mu',    [.3, 5, 0]*centi*poise, ...
                           'rho',   [1000, 700, 0]*kilogram/meter^3, ...
                           'n',     [2, 2, 0]);
c = 1e-5/barsa;
p_ref = 200*barsa;
fluid.bO = @(p) exp((p - p_ref)*c);

% saturation scaling
fluid.krPts  = struct('w', [0 0 1 1], 'ow', [0 0 1 1]);

scaling = {'SWL', .1, 'SWCR', .2, 'SWU', .9, 'SOWCR', .1, 'KRW', .9, 'KRO', .8};

%% setup wells
% 5-spot pattern
well_loc = load('well_loc_9_spots.mat');
well_loc = well_loc.well_loc;
W = [];
% for injector
num_inj = size(well_loc.inj_loc,1);
num_prod = size(well_loc.prod_loc,1);

for i = 1:num_inj
coord = sub2ind(grid_size,well_loc.inj_loc(i,1),well_loc.inj_loc(i,2));
W = addWell(W, G, rock, coord, 'Type' , 'rate', ...
                                   'Val'  , 300*meter^3/day, ...
                                   'Name' , sprintf('I%d', 1), ...
                                   'comp_i', [1 0], ...
                                   'Sign' , 1);
end


for i = 1:num_prod
coord = sub2ind(grid_size,well_loc.prod_loc(i,1),well_loc.prod_loc(i,2));
W = addWell(W, G, rock, coord, 'Type', 'bhp', ...
                                   'Val' , 100*barsa, ...
                                   'Name', sprintf('P%d', i), ...
                                   'comp_i', [0 1], ...
                                   'Sign', -1);
end

%% schedule of well

% begin the same control in the mrst example (setupModel2D)
% increase the original ts by 2 times
time_scale = 20;
ts = { [1 1 3 5 5 10 10 10 15 15 15 15 15 15 15]'*day*time_scale, ...
                   repmat(150/10, 10, 1)*day*time_scale, ...
                   repmat(150/6, 6, 1)*day*time_scale, ...
                   repmat(150/6, 6, 1)*day*time_scale};

       
numCnt = numel(ts);
[schedule.control(1:numCnt).W] = deal(W);
schedule.step.control = rldecode((1:4)', cellfun(@numel, ts));
schedule.step.val     = vertcat(ts{:});

%% GRAVITY
gravity on

%% initial state
state0 = initResSol(G, 200*barsa, [0, 1]); 

%% setup a reference model
model_ref = TwoPhaseOilWaterModel(G, rock, fluid);                       
model_ref = imposeRelpermScaling(model_ref, scaling{:});

%% get observation
% run ref model
[ws_ref, states_ref, r_ref] = simulateScheduleAD(state0, model_ref, schedule);

%% To-do add more observation later, here just return the saturation of the last time step
obs = states_ref{end}.s(:,1);

%% now save a .mat file for states_ref for later use

% first check whether states_ref.mat exists
if exist('states_ref.mat','file') == 2
    % if exists, delete it
    delete('states_ref.mat');
end

% save states_ref.mat
save('states_ref.mat','states_ref');

end