%% this matlab function is designed to be called in python via matlab.engine
% it receives a normalized square permeability map and returns the
% mismatch and its corresponding adjoint gradient



% the Norm_perm is normalized log(permeability) range in [0,1]

% always make sure the Norm_perm has the format (N,1)
Norm_perm = zeros(4096,1)
Norm_perm = reshape(Norm_perm,[],1);
MRST_folder = '/home1/junjieyu/mrst-2022b'
%% standard setup
% clear all
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
rock.perm = [permx permy permz];

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
well_loc = load('well_loc_5_spots.mat');
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
                                   'Val' , 150*barsa, ...
                                   'Name', sprintf('P%d', i), ...
                                   'comp_i', [0 1], ...
                                   'Sign', -1);
end

%% schedule of well

% begin the same control in the mrst example (setupModel2D)
% increase the original ts by 10 times
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

%% load ref_model
% check whether the states_ref exists, if not display error message
if exist('states_ref.mat','file') == 2
    load('states_ref.mat');
else
    error('states_ref.mat does not exist, please run the reference case first');
end

%% run the simulation and get the mismatch
model = GenericBlackOilModel(G, rock, fluid,'gas', false);
model = imposeRelpermScaling(model, scaling{:});
[ws, states, r] = simulateScheduleAD(state0, model, schedule);
% get the mismatch

% customize the weighting: 
% !!! TBD: what is a suitable appraoch to define the weight 
% approximately all the misfit will be within (0,1)* (dt/totTime*nnz(matchCases))
max_rate = 100*meter^3/day;
min_bhp = 100*barsa;
max_bhp = 200*barsa;

ww = 1/max_rate;
wo = 1/max_rate;
wp = 1/(max_bhp-min_bhp);

% compute misfit function value (first each summand corresonding to each time-step)
weighting =  {'WaterRateWeight',     ww, ...
              'OilRateWeight',       wo , ...
              'BHPWeight',           wp, ...
                'mismatchSum', false };
mismatch = matchObservedOW(model, states, schedule, states_ref, weighting{:});

% transform struct to matrix
% initalize the matrix as shape (3*num_wells, num_time_steps)
numsteps = size(schedule.step.val,1);
num_wells = num_inj + num_prod;
mismatch_mat = zeros(3*num_wells,numsteps);

for step = 1:numsteps
    mismatch_mat(:,step) = vertcat(mismatch{step});
end


% mismatch_qWs
mismatch_qWs = sum(sum(mismatch_mat(1:num_wells,:)));
% mismatch_qOs
mismatch_qOs = sum(sum(mismatch_mat(num_wells+1:2*num_wells,:)));
% mismatch_bhp
mismatch_bhp = sum(sum(mismatch_mat(2*num_wells+1:3*num_wells,:)));

mismatch_seperate = [mismatch_qWs;mismatch_qOs;mismatch_bhp];

mismatch_total = sum(mismatch_seperate);








