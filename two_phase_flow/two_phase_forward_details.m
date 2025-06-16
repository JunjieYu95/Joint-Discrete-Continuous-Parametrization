%% this matlab function is designed to be called in python via matlab.engine
% it receives a normalized square permeability map and returns the details results of the two-phase forward simulation, 
% includes well results states
% 

function [qWs, qOs, bhp, pressure, sw] = two_phase_forward_details(Norm_perm,MRST_folder)

    % the Norm_perm is normalized log(permeability) range in [0,1]
    
    % always make sure the Norm_perm has the format (N,1)
    Norm_perm = reshape(Norm_perm,[],1);
    
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
    
    % begin the same control in the mrst example (setupModel2D: 
    % path: /modules/optimization/examples/model2D/utils/setupModel2D.m)
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
    
    numsteps = size(schedule.step.val,1);
    num_welss = size(W,1);
    %% GRAVITY
    gravity on
    
    %% initial state
    state0 = initResSol(G, 200*barsa, [0, 1]);

    % boundary conditions
    % bc = pside([], G, 'YMin', 200*barsa);
    % bc = pside(bc, G, 'YMax', 200*barsa);
    % bc = pside(bc, G, 'XMin', 200*barsa);
    % bc = pside(bc, G, 'XMax', 200*barsa);
    
    %% run the simulation
    model = GenericBlackOilModel(G, rock, fluid,'gas', false);
    model = imposeRelpermScaling(model, scaling{:});
    [ws, states, r] = simulateScheduleAD(state0, model, schedule);

    % get well results from ws
    % initialize qWs, qOs, bhp
    qWs = zeros(num_welss ,numsteps); % each row represents one well, each column represents one time step
    qOs = zeros(num_welss ,numsteps);
    bhp = zeros(num_welss ,numsteps);

    for j = 1:numsteps
        qWs(:,j) = vertcat(ws{j}.qWs); % 
        % transform to m3/day
        qWs(:,j) = qWs(:,j)/meter^3*day;

        qOs(:,j) = vertcat(ws{j}.qOs);
        % transform to m3/day
        qOs(:,j) = qOs(:,j)/meter^3*day;

        bhp(:,j) = vertcat(ws{j}.bhp);
        % transform to barsa
        bhp(:,j) = bhp(:,j)/barsa;
    end

    % initialize pressure
    pressure = zeros(64*64, numsteps); % each row represents one flattened map, each column represents one time step
    % extract pressure from states
    for j = 1:numsteps
        pressure(:,j) = vertcat(states{j}.pressure);
        % transform to barsa
        pressure(:,j) = pressure(:,j)/barsa;
    end

    % initialize saturation
    sw = zeros(64*64, numsteps); % each row represents one flattened map, each column represents one time step
    % extract pressure from states
    for j = 1:numsteps
        sw(:,j) = vertcat(states{j}.s(:,1));
    end
    end
    
    
    
    
    
    
    
    