%% this matlab function is designed to be called in python via matlab.engine
% it receives a batch of normalized square permeability map (batch_size, 64, 64) and returns the
% gradient that calculated via central difference with parallel computing


function [mismatch_total_batch] = two_phase_forward_batch(Norm_perm_batch,MRST_folder)
    %% load ref_model
    % check whether the states_ref exists, if not display error message
    if exist('states_ref.mat','file') == 2
        temp = load('states_ref.mat');
        states_ref = temp.states_ref;
        % notice that if we directly load states_ref, it will have a vairable called states_ref
        % in the workspace, which will cause error in parfor loop since no such variable can be
        % broadcasted to workers

    else
        error('states_ref.mat does not exist, please run the reference case first');
    end
    %% standard setup
    % clear all
    MainDir = cd;
    cd(MRST_folder);NewPath = cd;cd(MainDir);
    addpath(genpath(NewPath));
    startup
    
    %% grid size
    nx = 64; ny = nx; nz = 1;
    grid_size = [nx,ny];
    Dx = 1000; Dy = 1000; Dz = 10;
    dx =  Dx/nx ; dy = Dy/ny; dz = Dz/nz ;
    G = cartGrid([nx, ny, nz],[Dx ,Dy ,Dz]);
    G = computeGeometry(G, 'Verbose', true);

    % in order to correctly load rock data within parfor loop, we need to create a rock_batch that
    % contains all the rock data required
    % initialize rock_batch

    min_perm = 50*milli*darcy;
    max_perm = 500*milli*darcy;
    well_loc = load('well_loc_9_spots.mat');
    well_loc = well_loc.well_loc;
    num_inj = size(well_loc.inj_loc,1);
    num_prod = size(well_loc.prod_loc,1);

    rock_batch = struct();
    W_batch = struct();
    schedule_batch = struct();

    for i = 1:size(Norm_perm_batch,1)
        % rock
        rock.poro = repmat(0.36 , [G.cells.num, 1]);
        Norm_perm = reshape(Norm_perm_batch(i,:,:),[],1);
        permx = exp(Norm_perm*(log(max_perm) - log(min_perm))+ log(min_perm));
        permy = permx;
        permz = 0.1*permx;
        rock.perm = [permx permy permz];
        rock_batch(i).rock = rock;

        %% W
        W = [];
        for ii = 1:num_inj
                coord = sub2ind(grid_size,well_loc.inj_loc(ii,1),well_loc.inj_loc(ii,2));
                W = addWell(W, G, rock, coord, 'Type' , 'rate', ...
                                                'Val'  , 300*meter^3/day, ...
                                                'Name' , sprintf('I%d', 1), ...
                                                'comp_i', [1 0], ...
                                                'Sign' , 1);
        end
        
        
        for ip = 1:num_prod
            coord = sub2ind(grid_size,well_loc.prod_loc(ip,1),well_loc.prod_loc(ip,2));
            W = addWell(W, G, rock, coord, 'Type', 'bhp', ...
                                            'Val' , 150*barsa, ...
                                            'Name', sprintf('P%d', i), ...
                                            'comp_i', [0 1], ...
                                            'Sign', -1);
        end
        W_batch(i).W = W;

        %% schedule
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
        schedule_batch(i).schedule = schedule;

    end
    % similarly we also need to create W_batch and schedule_batch
    % initialize W_batch


    % fluid
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

    %% schedule of well
    
    %% GRAVITY
    gravity on
    
    %% initial state
    state0 = initResSol(G, 200*barsa, [0, 1]);

    % initialize parallel pool
    batch_size = size(Norm_perm_batch,1);
    parpool(batch_size);
    % initialize the mismatch
    mismatch_total_batch = zeros(size(Norm_perm_batch,1),1);
    parfor par_i = 1:size(Norm_perm_batch,1)

        % the Norm_perm is normalized log(permeability) range in [0,1]
        
        rock = rock_batch(par_i).rock;
        W = W_batch(par_i).W;
        schedule = schedule_batch(par_i).schedule;

        %% run the simulation and get the mismatch
        model = GenericBlackOilModel(G, rock, fluid,'gas', false);
        model = imposeRelpermScaling(model, scaling{:});
        [ws, states, r] = simulateScheduleAD(state0, model, schedule);
        % get the mismatch
        
        max_rate = 100*meter^3/day;
        min_bhp = 100*barsa;
        max_bhp = 600*barsa;
        
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
        
        % update mismatch_total_batch
        mismatch_total_batch(par_i,1) = mismatch_total;

    end
    % delete the parallel pool: this is important or we will have error for next iteration
    delete(gcp('nocreate'));

    
    
    
    
    
    
    
    