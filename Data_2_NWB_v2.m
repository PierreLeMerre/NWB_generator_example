%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script that convert neuropixels, NIDaq and Facemap data into NWB format
%
% Developped by Pierre Le Merre
%
% Uses freezeColors Copyright (c) 2017, John Iversen for plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Your list of mice
Mouse_list = {'128514','128515','128516','147463','147465',... 
    '152414','152417','152419','156130','156131',... 
    '216300','216301','225757','225758','225759',... 
    '258412','258414','258416','258419','259112',... 
    '268947','268951','273853','273855','273858'};   


for M = 1 : numel(Mouse_list) % loop over mice


clearvars -except Mouse_list M

%% Main Parameters
% Parameters
Mouse = Mouse_list{M};
probe = '-probe0'; % '' if no multiple probes
rec_day = '';
disp(['Processing Mouse ' Mouse probe])
Brain_Region = 'mPFC'; %'mPFC','Aud'
Task = 'Aversion';

% Raw Data identifiers
if strcmp(Mouse,'PL026')
    glk_id = 6; % Spike GLX identifier
    imec_nb = 0; %imec number
elseif strcmp(Mouse,'PL035')
    glk_id = 3; % Spike GLX identifier
    imec_nb = 1; %imec number
elseif strcmp(Mouse,'PL036')
    glk_id = 0; % Spike GLX identifier
    imec_nb = 1; %imec number
else
    glk_id = 0; % Spike GLX identifier
    imec_nb = 0; %imec number
end

% Get the correct Mouse identifier from our local mouse Database (Tick@lab)
if strcmp(Mouse,'242820') || strcmp(Mouse,'249939') || strcmp(Mouse,'249940') || ...
        strcmp(Mouse,'250252') || strcmp(Mouse,'250253') || strcmp(Mouse,'250256')
    probe_nb = M + 57;
else
    probe_nb = M + 16; 
end


%% Last chance to remove some units (for any reason, light artifacts, bad spike sorting,....)
Unit_IDX2rmv = [];
if strcmp(Mouse,'258414') % light artifacts
    Unit_IDX2rmv = [195 160 79];
elseif strcmp(Mouse,'225759') % light artifacts
    Unit_IDX2rmv = [390 355 352 351 350 348 329 327 325 319 312 311 303 301 253 252 196 193 159];
end

%% Loading of the data structures
% Meta data on server
[meta_ap] = ReadMeta([Mouse '_g' num2str(glk_id) '_t0.imec' num2str(imec_nb) '.ap.meta'], ['/Volumes/projects/From_Home_Le_Merre/DATA/' Mouse '/Recordings/' Mouse '_g' num2str(glk_id) '/' Mouse '_g' num2str(glk_id) '_imec' num2str(imec_nb) '/']);
[meta_lfp] = ReadMeta([Mouse '_g' num2str(glk_id) '_t0.imec' num2str(imec_nb) '.lf.meta'], ['/Volumes/projects/From_Home_Le_Merre/DATA/' Mouse '/Recordings/' Mouse '_g' num2str(glk_id) '/' Mouse '_g' num2str(glk_id) '_imec' num2str(imec_nb) '/']);
[meta_nidaq] = ReadMeta([Mouse '_g' num2str(glk_id) '_t0.nidq.meta'], ['/Volumes/projects/From_Home_Le_Merre/DATA/' Mouse '/Recordings/' Mouse '_g' num2str(glk_id) '/']);

% Identifier
session = meta_ap.fileCreateTime(1:end-9);
session(regexp(session,'-'))=[];
identifier = [Mouse '_' session probe]; % unique identifier for saving NWB.

% Processed Spike data (Good units)
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/Spikes_data.mat'])

% Processed LFP data (Downsampled)
load(['/Volumes/projects/From_Home_Le_Merre/NPX_LFP/' Mouse probe '/Lfp_data_all.mat'])

% Extracted EMG data
load(['/Users/pielem/Desktop/ANALYSIS/' Brain_Region '_' Task '_Neuropixels/EMG/' Mouse probe '/emg.mat'])

% Local Trial data
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/trial_ts.mat'])
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/Trials.mat'])

% Local Sound data
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/sound_ts.mat'])

% FaceMap Pupil Data
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/face/pupil_area.mat'])
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/face/blink_trace.mat'])
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/face/video_ts.mat'])

% FaceMap Face Data
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/face/face1.mat'])
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/face/face2.mat'])
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/face/face3.mat'])

% Few Quality metrics
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/sorting_quality/sorting_metrics.mat'])

% Probe reconstruction data and histology (Sharp Track + csv conversion)
regions = importdata(['/Users/pielem/Desktop/NWB_conversion/' Brain_Region '-Probes_csv/probe' num2str(probe_nb) '_sites.csv']);

% Tick@lab database (local mouse husbandry system to get basic subject info)
T = readtable(['/Users/pielem/Desktop/NWB_conversion/' Brain_Region '-Tick@lab_data.xls']);

% Additional timestamps
% Air Puff
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/airpuff_ts.mat'])

% Optogenetic Data
load(['/Volumes/labs/pielem/DATA/' Brain_Region '_' Task '/' Mouse probe '/opto_ts.mat'])


%% Create nwb Structure

nwb = NwbFile( ...
    'session_description', 'Neuropixels recording in mPFC during aversive conditioning task',...
    'identifier', identifier, ...
    'session_start_time', meta_ap.fileCreateTime, ...
    'general_experiment_description', 'Neuropixels recording in mPFC during aversive conditioning task',...
    'general_experimenter', 'Le Merre, Pierre', ... % optional
    'general_lab', 'Carlen Lab',...
    'general_keywords', {'Prefrontal Cortex', 'Auditory Processing', 'Aversion'},...
    'general_session_id', identifier, ... % optional
    'general_institution', 'Karolinska Institutet',... % optional
    'general_stimulus',{'pure tone, blue noise, photostim, air puff'}, ...
    'general_surgery', ['Mice were prepared for ',...
    'electrophysiology and optogenetics with a recording chamber, a headpost and optic fibers. ',...
    'The scalp and periosteum over the dorsal surface of the skull were removed. ',...
    'A layer of cyanoacrylate adhesive (Locite) ',...
    'was directly applied to the intact skull. A custom made headpost ',...
    ' and optic fibers were placed on the skull ',...
    '(headpost: approximately over cerebellum and right hemisphere; optic fibers: over LHA) and cemented in place ',...
    'with Palavit/Paladur.'
    ]);


%% Subject information
split_age = regexp(T.Age{probe_nb},'\d*','Match');
if strcmp(T.S{probe_nb},'m')
    sex = 'M';
elseif strcmp(T.S{probe_nb},'f')
    sex = 'F';
end

subject = types.core.Subject( ...
    'subject_id', Mouse, ...
    'date_of_birth', T.DoB(probe_nb),...
    'age', ['P' split_age{1} 'M' split_age{2} 'D'], ...
    'description', 'Mouse implanted with a chamber for acute recordings', ...
    'species', 'Mus musculus', ...
    'genotype', T.BreedingLine{probe_nb}, ...
    'sex', sex);
nwb.general_subject = subject;

%% Trials

    % A bit of reorganization (Make logical vectors, exepctions, etc...)
    sound_bol = zeros(1,numel(trial_ts.ts_corr));
    pure_bol= zeros(1,numel(trial_ts.ts_corr));
    blue_bol= zeros(1,numel(trial_ts.ts_corr));
    Freq_s= zeros(1,numel(trial_ts.ts_corr));
    Amp= zeros(1,numel(trial_ts.ts_corr));

    opto_bol = zeros(1,numel(trial_ts.ts_corr));
    Freq_o = zeros(1,numel(trial_ts.ts_corr));
    Pow = zeros(1,numel(trial_ts.ts_corr));
    pulse_dur = zeros(1,numel(trial_ts.ts_corr));
    pulse_ts = nan(numel(trial_ts.ts_corr),20);

    air_bol = zeros(1,numel(trial_ts.ts_corr));
    pulse_dur_air = zeros(1,numel(trial_ts.ts_corr));
    pulse_amp_air = zeros(1,numel(trial_ts.ts_corr));

    s_cnt = 1;
    o_cnt = 1;
    idx2remove = [];
    for i = 1: numel(trial_ts.ts_corr)
        if any(trial_ts.ts_corr(i) == sound_ts.ts_corr)
            sound_bol(i) = 1;
            if sound_ts.soundtype(s_cnt)==-1
                pure_bol(i) = 1;
            elseif sound_ts.soundtype(s_cnt)==1
                blue_bol(i) = 1;
            end
            Freq_s(i) = sound_ts.freq(s_cnt);
            Amp(i) = sound_ts.amp(s_cnt);
            s_cnt = s_cnt + 1;
        end

        if  min(abs(opto_ts.ts_on_corr/opto_ts.sr-trial_ts.ts_corr(i)/trial_ts.sr))<0.8
            opto_bol(i) = 1;
            if 0<Trials(i) && Trials(i)<5
                Freq_o(i) = 40;
                pulse_dur(i) = 0.005;
                pulse_ts(i,:) = opto_ts.ts_on_corr(o_cnt:o_cnt+19);
                o_cnt = o_cnt + 20;
            elseif Trials(i)==5
                Freq_o(i) = 40;
                pulse_dur(i) = 0.005;
                pulse_ts(i,:) = opto_ts.ts_on_corr(o_cnt:o_cnt+19);
                o_cnt = o_cnt + 20;
            elseif Trials(i)==6
                Freq_o(i) = 40;
                pulse_dur(i) = 0.005;
                pulse_ts(i,:) = opto_ts.ts_on_corr(o_cnt:o_cnt+19);
                o_cnt = o_cnt + 20;
            elseif Trials(i)==7
                Freq_o(i) = 40;
                pulse_dur(i) = 0.005;
                pulse_ts(i,:) = opto_ts.ts_on_corr(o_cnt:o_cnt+19);
                o_cnt = o_cnt + 20;
            elseif Trials(i)==8
                Freq_o(i) = 40;
                pulse_dur(i) = 0.005;
                if strcmp(identifier,'152417_20191023-probe0') && o_cnt==6021
                    pulse_ts(i,1:19) = opto_ts.ts_on_corr(o_cnt:o_cnt+18);
                    pulse_ts(i,20) = NaN;
                else
                    pulse_ts(i,:) = opto_ts.ts_on_corr(o_cnt:o_cnt+19);
                end
                o_cnt = o_cnt + 20;
            elseif Trials(i)==9
                Freq_o(i) = 5;
                pulse_dur(i) = 0.010;
                pulse_ts(i,:) = [opto_ts.ts_on_corr(o_cnt:o_cnt+2); nan(17,1)];
                o_cnt = o_cnt + 3;
            elseif Trials(i)==10
                Freq_o(i) = 10;
                pulse_dur(i) = 0.005;
                pulse_ts(i,:) = [opto_ts.ts_on_corr(o_cnt:o_cnt+4); nan(15,1)];
                o_cnt = o_cnt + 5;
            elseif Trials(i)==11
                Freq_o(i) = 20;
                pulse_dur(i) = 0.005;
                pulse_ts(i,:) = [opto_ts.ts_on_corr(o_cnt:o_cnt+9); nan(10,1)];
                o_cnt = o_cnt + 10;
            elseif Trials(i)==12
                Freq_o(i) = 1;
                pulse_dur(i) = 1;
                pulse_ts(i,:) = [opto_ts.ts_on_corr(o_cnt) ; nan(19,1)];
                o_cnt = o_cnt + 1;
            elseif Trials(i)==0
                idx2remove = [idx2remove i];
                Freq_o(i) = nan;
                pulse_dur(i) = nan;
                pulse_ts(i,:) = [nan ; nan(19,1)];
                o_cnt = o_cnt + 1;
            end

        end

        if  min(abs(airpuff_ts.ts_corr/airpuff_ts.sr-trial_ts.ts_corr(i)/trial_ts.sr))<0.8
            air_bol(i) = 1;
            pulse_dur_air(i) = 0.02;
            pulse_amp_air(i) = 5;
        end
    end
    opto_bol(end-30:end)=1;

    Freq_o(idx2remove) = [];
    Pow(idx2remove) = [];
    pulse_dur(idx2remove) = [];
    pulse_ts(idx2remove,:) = [];
    sound_bol(idx2remove) = [];
    pure_bol(idx2remove) = [];
    blue_bol(idx2remove) = [];
    Freq_s(idx2remove) = [];
    Amp(idx2remove) = [];
    opto_bol(idx2remove) = [];
    air_bol(idx2remove) = [];
    pulse_dur_air(idx2remove) = [];
    pulse_amp_air(idx2remove) = [];
    trial_ts.ts_corr(idx2remove) = [];
    Trials(idx2remove) = [];


    % Finally adding it to the nwb sturcture
    trials = types.core.TimeIntervals( ...
        'colnames', {'start_time', 'stop_time',...
        'Block',...
        'sound','sound_puretone','sound_bluenoise','sound_freq','sound_amp',...
        'optogenetics','opto_freq','opto_power','opto_pulse_duration','opto_pulse_ts',...
        'airpuff','airpuff_duration','airpuff_amp'}, ...
        'description', 'trial data and properties', ...
        'id', types.hdmf_common.ElementIdentifiers('data', 0:numel(trial_ts.ts_corr)-1), ... %0 indexed
        'start_time', types.hdmf_common.VectorData('data', trial_ts.ts_corr./trial_ts.sr, 'description','start time of trial'), ...
        'stop_time', types.hdmf_common.VectorData('data', trial_ts.ts_corr./trial_ts.sr+2, 'description','end of each trial'),...
        'Block',types.hdmf_common.VectorData('data', Trials', 'description','Block number'),...
        'sound',types.hdmf_common.VectorData('data', sound_bol','description','bolean for sound'),...
        'sound_puretone',types.hdmf_common.VectorData('data', sound_bol','description','bolean for sound'),...
        'sound_bluenoise',types.hdmf_common.VectorData('data', blue_bol','description','bolean for blue noise'),...
        'sound_freq',types.hdmf_common.VectorData('data', Freq_s','description','sound frequency'),...
        'sound_amp',types.hdmf_common.VectorData('data', Amp','description','sound amplitude'),...
        'optogenetics',types.hdmf_common.VectorData('data', logical(opto_bol'),'description','bolean for optogenetics'),...
        'opto_freq',types.hdmf_common.VectorData('data', Freq_o','description','pulse train frequency'),...
        'opto_power',types.hdmf_common.VectorData('data', Pow','description','laser power'),...
        'opto_pulse_duration',types.hdmf_common.VectorData('data', pulse_dur','description','pulse duration (s)'),...
        'opto_pulse_ts',types.hdmf_common.VectorData('data', pulse_ts','description','individual pulses timestamps'),...
        'airpuff',types.hdmf_common.VectorData('data', air_bol','description','bolean for airpuff'),...
        'airpuff_duration',types.hdmf_common.VectorData('data', pulse_dur_air','description','Airpuff duration (s)'),...
        'airpuff_amp',types.hdmf_common.VectorData('data', pulse_amp_air','description','Airpuff volatge control amplitude (V)'));

    nwb.intervals_trials = trials;

 


%% Stimuli and Optogenetic timestamps

    % Auditory Stimuli
    AudStim_10kHz = types.core.TimeSeries(...
        'data', sound_ts.amp(sound_ts.freq==10000), ...
        'data_unit', 'Volts', ...
        'description', 'Sound Amplitude in V corresponding to different sound pressures', ...
        'timestamps', sound_ts.ts_corr(sound_ts.freq==10000)./sound_ts.sr, ...
        'timestamps_unit', 's');
    nwb.stimulus_presentation.set('AudStim_10kHz', AudStim_10kHz);
    AudStim_10kHz_ref = types.untyped.ObjectView('/stimulus/presentation/AudStim_10kHz');

    AudStim_bluenoise = types.core.TimeSeries(...
        'data', sound_ts.amp(sound_ts.soundtype==1), ...
        'data_unit', 'Volts', ...
        'description', 'Sound Amplitude in V corresponding to different sound pressures', ...
        'timestamps', sound_ts.ts_corr(sound_ts.soundtype==1)./sound_ts.sr, ...
        'timestamps_unit', 's');
    nwb.stimulus_presentation.set('passive_bluenoise', AudStim_bluenoise);
    AudStim_bluenoise_ref = types.untyped.ObjectView('/stimulus/presentation/AudStim_bluenoise');

    % Air Puffs
    air_puff = types.core.TimeSeries(...
        'data', 5*ones(1,numel(airpuff_ts.ts_corr)), ...
        'data_unit', 'Volts', ...
        'description', 'Air Puff in V corresponding to TTL', ...
        'timestamps', airpuff_ts.ts_corr./airpuff_ts.sr, ...
        'timestamps_unit', 's');
    nwb.stimulus_presentation.set('air_puff', air_puff);
    air_puff_ref = types.untyped.ObjectView('/stimulus/presentation/air_puff');


    % Optogenetic pulse
    % device
    laser = types.core.Device(...
        'description', 'laser-473nm', ...
        'manufacturer', 'Cobolt Inc., Cobolt 06 MLD');
    nwb.general_devices.set('laser', laser);

    % stimulus site
    ogen_stim_site = types.core.OptogeneticStimulusSite( ...
        'device', types.untyped.SoftLink(laser), ...
        'description', 'PhotoStimulation (ChR2) of VGlut2 positive LHA-LHb projecting neurons; AP -1.1 ML 1.1 DV -4.5 mm', ...
        'excitation_lambda', 473.0, ...
        'location', 'LHA (Lateral Hypothalamic Area)');

    nwb.general_optogenetics.set('OptogeneticStimulusSite', ogen_stim_site);

    % Adding Timeseries to have power values
    ogen_series = types.core.OptogeneticSeries( ...
        'description', 'Optogenetic stim pulse data: pulse of 5ms with 1ms SinRamp in and out, Optical power in data (mW)', ...
        'data', opto_ts.amp, ...
        'data_unit', 'mW', ...
        'site', types.untyped.SoftLink(ogen_stim_site), ...
        'timestamps', opto_ts.ts_on_corr./opto_ts.sr, ...
        'timestamps_unit', 's');

    nwb.stimulus_presentation.set('OptogeneticSeries', ogen_series);


%% Extracellular ephys data

nprobes = 1;
nchannels_per_probe = 383; %32;
variables = {'AP', 'DV', 'ML', 'imp', 'location', 'filtering', 'group', 'group_name','label'}; %, 'axonal density'
tbl = cell2table(cell(0, length(variables)), 'VariableNames', variables);
device = types.core.Device(...
    'description', 'Neuropixels 3B1', ...
    'manufacturer', 'imec');
nwb.general_devices.set('array', device);


    for iprobe = 1:nprobes
        electrode_group = types.core.ElectrodeGroup( ...
            'description', ['electrode group for probe' num2str(iprobe)], ...
            'location', 'mPFC', ...
            'device', types.untyped.SoftLink(device) ...
            );
        nwb.general_extracellular_ephys.set(['probe' num2str(iprobe)], electrode_group);
        group_object_view = types.untyped.ObjectView(electrode_group);


        for ielec = 1:nchannels_per_probe

            tbl = [tbl; ...
                {regions.data(ielec,1)/100,... 'AP'
                -1.*regions.data(ielec,2)/100,... 'DV'
                regions.data(ielec,3)/100,... 'ML'
                150000,... 'imp'
                regions.textdata{ielec},... 'location'
                '0.3-10_kHz_and_0.5-500_Hz', ... 'filtering'  %NaN,... 'axonal density'
                group_object_view, ... 'group'
                ['probe' num2str(iprobe)], ....
                ['probe' num2str(iprobe) 'elec' num2str(ielec)]}]; %'label'


        end

    end


electrode_table = util.table2nwb(tbl, 'all electrodes');
nwb.general_extracellular_ephys_electrodes = electrode_table;

% Electrical serie to store voltage data
electrodes_object_view = types.untyped.ObjectView( ...
    '/general/extracellular_ephys/electrodes');
electrode_table_region = types.hdmf_common.DynamicTableRegion( ...
    'table', electrodes_object_view, ...
    'description', 'all electrodes', ...
    'data', [0:size(tbl,1)-1]'); % 0 indexed!! -1

electrodes_object_view2 = types.untyped.ObjectView( ...
    '/general/extracellular_ephys/electrodes');
electrode_table_region2 = types.hdmf_common.DynamicTableRegion( ...
    'table', electrodes_object_view2, ...
    'description', 'EMG electrode', ...
    'data', [0]'); % 0 indexed!!

% LFP electrical serie
% compression
lfp_comp = types.untyped.DataPipe('data', Lfp_data.Channel_Samples);
% store
electrical_series = types.core.ElectricalSeries( ...
    'starting_time', 0.0, ... % seconds
    'starting_time_rate', Lfp_data.Session_SR(1), ... %str2double(meta_lfp.imSampRate), ... % Hz
    'data', lfp_comp, ...
    'electrodes', electrode_table_region, ...  %% correct for all electrodes...not good right now
    'data_unit', 'volts',...
    'description','LFP');

% Finaly store the downsampled LFP data into ecephys processing module
ecephys_module = types.core.ProcessingModule(...
    'description', 'extracellular electrophysiology');
ecephys_module.nwbdatainterface.set('LFP', types.core.LFP( ...
    'ElectricalSeries', electrical_series));
nwb.processing.set('ecephys', ecephys_module);

% EMG electrical serie
EMG_electrical_series = types.core.ElectricalSeries( ...
    'starting_time', 0.0, ... % seconds
    'starting_time_rate', 1000, ... % Hz
    'data', emg, ...
    'electrodes', electrode_table_region2, ...
    'data_unit', 'volts',...
    'description','EMG');

% Store the EMG data into ecephys processing module
ecephys_module.nwbdatainterface.set('EMG', types.core.LFP( ...
    'ElectricalSeries', EMG_electrical_series));
nwb.processing.set('ecephys', ecephys_module);


%% Spike data

% remove units from our matlablab structure first
if ~isempty(Unit_IDX2rmv)
    for ui = 1 : numel(Unit_IDX2rmv)
        uidx = Unit_IDX2rmv(ui);

        % Empty spike struct
        Spikes_data.Mouse_Name(uidx,:) = [];
        Spikes_data.Mouse_Genotype(uidx,:) = [];
        Spikes_data.Mouse_Sex(uidx,:) = [];
        Spikes_data.Mouse_DateOfBirth(uidx,:) = [];

        Spikes_data.Session_Counter(uidx) = [];
        Spikes_data.Session_StartTime(uidx,:) = [];
        Spikes_data.Session_Type(uidx,:) = [];
        Spikes_data.Session_Probe(uidx,:) = [];
        Spikes_data.Session_SR(uidx) = [];

        Spikes_data.Cell_Number(uidx) = [];
        Spikes_data.Cell_Cluster(uidx) = [];
        Spikes_data.Cell_Channel(uidx) = [];
        Spikes_data.Cell_Depth(uidx) = [];
        Spikes_data.Cell_Coordinates(uidx,:) = [];
        Spikes_data.Cell_Region(uidx) = [];
        Spikes_data.Cell_RegionFullName(uidx) = [];
        Spikes_data.Cell_SpikeTimes(uidx) = [];
        Spikes_data.Cell_MainSpikeWaveforms(uidx,:,:) = [];
        Spikes_data.Cell_MainSpikeWaveform(uidx,:) = [];
        Spikes_data.Cell_SpikeTemplate(uidx,:) = [];
        Spikes_data.Cell_AutoCorrelogram(uidx) = [];

        Spikes_data.Cell_spike_orientation(uidx) = [];
        Spikes_data.Cell_spike_height(uidx) = [];
        Spikes_data.Cell_spike_Peak_to_Trough(uidx) = [];
        Spikes_data.Cell_spike_PTR(uidx) = [];
        Spikes_data.Cell_spike_HW(uidx) = [];
        Spikes_data.Cell_spike_duration(uidx) = [];
        Spikes_data.Cell_spike_PeaktoOnset(uidx) = [];
        Spikes_data.Cell_spike_PeaktoBaseline(uidx) = [];
        Spikes_data.Cell_spike_PeaktoBaselineArea(uidx) = [];
        Spikes_data.Cell_spike_TroughSize(uidx) = [];
        Spikes_data.Cell_spike_PeaktoOnsetTime(uidx) = [];
        Spikes_data.Cell_spike_dVdT(uidx) = [];

        Spikes_data.Cell_spike_FS_ID(uidx) = [];
        Spikes_data.Cell_spike_RS_ID(uidx) = [];
        Spikes_data.Cell_spike_FR(uidx) = [];
        Spikes_data.Cell_spike_CV(uidx) = [];


        % Empty Cell metrics
        sorting_metrics.ISIviolations(uidx) = [];
        sorting_metrics.fpRate(uidx) = [];
        sorting_metrics.UnitQuality(uidx) = [];
        sorting_metrics.ContaminationRate(uidx) = [];
    end
end


%reformat spike data
num_cells = size(Spikes_data.Cell_SpikeTimes,1);
spikes = cell(1, num_cells);
channels = cell(1, num_cells);
for iunit = 1:num_cells
    spikes{iunit} = [];
    spikes{iunit} = double(Spikes_data.Cell_SpikeTimes{iunit,1}{1,1})./Spikes_data.Session_SR(iunit);
    channels{iunit} = [];
    channels{iunit} = ones(1,length(spikes{iunit})).*Spikes_data.Cell_Channel(iunit);
end

% get waveforms
wf = Spikes_data.Cell_MainSpikeWaveform';
sr = Spikes_data.Session_SR(1);

% Calculate some metrics on the fly: Spike height, peak to trough ratio, peak to trough duration,
% half width, etc....
disp(['Calculating a few spike metrics for file ' identifier '.nwb']);
firstquarter_idx=ceil(size(wf,1)/4);
mid_idx=round(size(wf,1)/2);

wf_metrics = [];
spike_Orientation = nan(1,size(wf,2));
spike_PeaktoOnsetTime = nan(1,size(wf,2));
spike_PeaktoOnset = nan(1,size(wf,2));
spike_duration = nan(1,size(wf,2));
spike_HW = nan(1,size(wf,2));
spike_height = nan(1,size(wf,2));
spike_PTR = nan(1,size(wf,2));
spike_Peak_to_Trough = nan(1,size(wf,2));
spike_PeaktoBaseline = nan(1,size(wf,2));
spike_PeaktoBaselineArea = nan(1,size(wf,2));
spike_TroughSize = nan(1,size(wf,2));
spike_dVdT = nan(1,size(wf,2));

for i=1:size(wf,2)

    spike_shape=wf(:,i)';
    spike_shape=spike_shape-mean(spike_shape(1:round(mid_idx/2)));

    % get Spike orientation
    if (abs(min(spike_shape)-mean(spike_shape(1:mid_idx)))) >= (abs(max(spike_shape)-mean(spike_shape(1:mid_idx))))
        spike_Orientation(i) = -1;
        peakindex=find(spike_shape==min(spike_shape(firstquarter_idx:end)));
        Troughindex=find(spike_shape(peakindex+1:end)==max(spike_shape(peakindex+1:end)))+peakindex;

    elseif (abs(min(spike_shape)-mean(spike_shape(1:mid_idx)))) < (abs(max(spike_shape)-mean(spike_shape(1:mid_idx))))
        spike_Orientation(i) = 1;
        peakindex=find(spike_shape==max(spike_shape(firstquarter_idx:end)));
        Troughindex=find(spike_shape(peakindex+1:end)==min(spike_shape(peakindex+1:end)))+peakindex;
    end

    % get Peak onset time and Peak onset
    onsetindex=find(abs(spike_shape(1:peakindex))<=.1*abs(spike_shape(peakindex)),1,'last');
    spike_PeaktoOnsetTime(i)=(peakindex-onsetindex)./sr; % in ms
    spike_PeaktoOnset(i)=abs(spike_shape(peakindex)-spike_shape(onsetindex)); % in uv
    index33=find(abs(spike_shape(peakindex:end)-spike_shape(onsetindex))<=.33*spike_PeaktoOnset(i),1,'first')+peakindex;
    if isempty(index33)
        index33=numel(spike_shape);
    end

    % get spike duration
    spike_duration(i)=(index33-peakindex)./sr; % in ms

    % get half width
    changeindex_1=find(abs(spike_shape(1:peakindex))<=.5*abs(spike_shape(peakindex)),1,'last');
    changeindex_2=find(abs(spike_shape(peakindex:end))<=.5*abs(spike_shape(peakindex)),1,'first')+peakindex;
    if isempty(changeindex_2)
        changeindex_2=numel(spike_shape);
    end
    spike_HW(i)=(changeindex_2-changeindex_1)./sr; % in ms

    % get spike height
    spike_height(i)=-spike_shape(peakindex); % in uv

    % get peak to through ratio
    if ~isempty(spike_shape(Troughindex))
        spike_PTR(i)=spike_shape(Troughindex)/spike_shape(peakindex); % trough to peak ratio
    else
        spike_PTR(i)=NaN;
    end

    % get peak to baseline, peak to baseline area, through size and dV/dT
    if peakindex ~= numel(spike_shape)
        spike_Peak_to_Trough(i)=abs(peakindex-Troughindex)./sr; %  peak to trough in ms
        BacktoBaselineIndex=find(abs(spike_shape(peakindex:end))<=.05*abs(spike_shape(peakindex)),1,'first')+peakindex-1;
    elseif peakindex == numel(spike_shape)
        spike_Peak_to_Trough(i)=NaN; %  peak to trough in ms
        BacktoBaselineIndex=numel(spike_shape);
    end

    if isempty(BacktoBaselineIndex)
        BacktoBaselineIndex=numel(spike_shape);
    end

    if peakindex ~= numel(spike_shape)
        spike_PeaktoBaseline(i)=(BacktoBaselineIndex-peakindex)/sr; % in ms
        spike_PeaktoBaselineArea(i)=trapz(spike_shape(peakindex:BacktoBaselineIndex));
        spike_TroughSize(i)=max(spike_shape(peakindex:end));
        spike_dVdT(i)=max(abs(diff(spike_shape(onsetindex:peakindex))))/max(abs(diff(spike_shape(peakindex:BacktoBaselineIndex))));
    elseif peakindex == numel(spike_shape)
        spike_PeaktoBaseline(i)=NaN; % in ms
        spike_PeaktoBaselineArea(i)=NaN;
        spike_TroughSize(i)=NaN;
        spike_dVdT(i)=NaN;
    end


end

wf_metrics.spike_orientation=spike_Orientation;
wf_metrics.spike_height=spike_height;
wf_metrics.spike_Peak_to_Trough=spike_Peak_to_Trough;
wf_metrics.spike_PTR=spike_PTR;
wf_metrics.spike_HW=spike_HW;
wf_metrics.spike_duration=spike_duration;
wf_metrics.spike_PeaktoOnset=spike_PeaktoOnset;
wf_metrics.spike_PeaktoBaseline=spike_PeaktoBaseline;
wf_metrics.spike_PeaktoBaselineArea=spike_PeaktoBaselineArea;
wf_metrics.spike_TroughSize=spike_TroughSize;
wf_metrics.spike_PeaktoOnsetTime=spike_PeaktoOnsetTime;
wf_metrics.spike_dVdT=spike_dVdT;


% Split FS and RS
disp(['Split RSU and FSU for file ' identifier '.nwb']);

% Hardcoded values working for this Neuropixels data
low_tshld = 0.400;%in ms
high_tshld = 0.440;%in ms

A=wf_metrics.spike_Peak_to_Trough(wf_metrics.spike_Peak_to_Trough<=low_tshld); %low threshold
B=wf_metrics.spike_Peak_to_Trough(wf_metrics.spike_Peak_to_Trough>low_tshld & wf_metrics.spike_Peak_to_Trough<high_tshld);
C=wf_metrics.spike_Peak_to_Trough(wf_metrics.spike_Peak_to_Trough>=high_tshld); %high threshold

% Display the classification
figure;
histogram(A,0:.02:1.5,'FaceColor','r')
hold on
histogram(C,0:.02:1.5,'FaceColor','b')
histogram(B,0:.02:1.5,'FaceColor',[.5 .5 .5])
title(['Peak to Trough Histogram, ' identifier])
xlabel('Peak to Trough (ms)');
ylabel('counts');
legend('Fast Spiking','Regular Spiking','Location','northeast')
line([low_tshld low_tshld],[0 50],'LineWidth',1,'Color',[0 0 0])
line([high_tshld high_tshld],[0 50],'LineWidth',1,'Color',[0 0 0])

FS_id = zeros(1,size(wf,2));
RS_id = zeros(1,size(wf,2));

FS_id(wf_metrics.spike_Peak_to_Trough<=low_tshld) = 1;
RS_id(wf_metrics.spike_Peak_to_Trough>=high_tshld) = 1;

% Quality metrics
All_metrics = [sorting_metrics.ISIviolations' sorting_metrics.fpRate'];

% Write spike data in nwb
disp('Add spike metrics and WS-NS classification to nwb file...')


% Electrodes
electrodes_object_view = types.untyped.ObjectView('/general/extracellular_ephys/electrodes');

% Exception case for one mouse with less channels
if strcmp(identifier,'147463_20191113-probe0') %%100 channels cut form this recording
    [electrodes, electrodes_index] = util.create_indexed_column(num2cell(int64(Spikes_data.Cell_Channel-1+100)),'/units/electrodes'); 
else
    [electrodes, electrodes_index] = util.create_indexed_column(num2cell(int64(Spikes_data.Cell_Channel-1)),'/units/electrodes'); 

end

% Spike times
[spike_times_vector, spike_times_index] = util.create_indexed_column(spikes,'spike times');


% Waveforms
waveform_mean = types.hdmf_common.VectorData('data',wf,'description', 'mean of waveform');

% Filling the Unit structure
nwb.units = types.core.Units( ...
    'colnames', {'spike_times', 'electrodes','waveform_means',... % Fooling pynwb while using 'waveform_meanS' so we don't get an error....
    'quality','PeaktoOnsetTime','PeaktoOnset','duration','HW','height','PTR','Peak_to_Trough',...
    'PeaktoBaseline','PeaktoBaselineArea','TroughSize','dVdT','FS','RS'}, ...
    'description', 'units table', ...
    'id', types.hdmf_common.ElementIdentifiers('data', int64(0:length(spikes) - 1)),...
    'spike_times',spike_times_vector,...
    'spike_times_index',spike_times_index,...
    'electrodes', types.hdmf_common.DynamicTableRegion('table', electrodes_object_view, 'description', 'electrode of the Main Waveform', 'data', electrodes.data),...
    'waveform_means', waveform_mean,...
    'quality',types.hdmf_common.VectorData('data', All_metrics','description', 'ISI violation, false positive rate'),...
    'PeaktoOnsetTime',types.hdmf_common.VectorData('data', spike_PeaktoOnsetTime,'description', 'Peak to Onset Time (ms)'),...
    'PeaktoOnset',types.hdmf_common.VectorData('data', spike_PeaktoOnset,'description', 'Peak to Onset (uV)'),...
    'duration',types.hdmf_common.VectorData('data', spike_duration,'description', 'duration (ms)'),...
    'HW',types.hdmf_common.VectorData('data', spike_HW,'description', 'Half width (ms)'), ...
    'height',types.hdmf_common.VectorData('data', spike_height,'description', 'Height (uV)'),...
    'PTR',types.hdmf_common.VectorData('data', spike_PTR,'description', 'Peak to trough ratio'),...
    'Peak_to_Trough',types.hdmf_common.VectorData('data', spike_Peak_to_Trough,'description', 'Peak to Trough (ms)'),...
    'PeaktoBaseline',types.hdmf_common.VectorData('data', spike_PeaktoBaseline,'description', 'Peak to Baseline (ms)'),...
    'PeaktoBaselineArea',types.hdmf_common.VectorData('data', spike_PeaktoBaselineArea,'description', 'Peak to Baseline Area'),...
    'TroughSize',types.hdmf_common.VectorData('data', spike_TroughSize,'description', 'TroughSize (uV)'),...
    'dVdT',types.hdmf_common.VectorData('data', spike_dVdT,'description', 'dVdT (uV.ms-1)'),...
    'FS',types.hdmf_common.VectorData('data', logical(FS_id),'description', 'logical vector for fast spiking units'),...
    'RS',types.hdmf_common.VectorData('data', logical(RS_id),'description', 'logical vector for regular spiking units'));


%% Pupil data

%Pupil behavioral Timeserie
eye_area = types.core.TimeSeries( ...
    'data', pupil_area,...
    'data_unit', 'pixel^2',...
    'timestamps',ts_video,...
    'timestamps_unit', 's',...
    'description', 'smoothed pupil trace extracted with Facemap algorithm form the pupil movies');

%% Blink data
% blink behavioral Timeserie
blink = types.core.TimeSeries( ...
    'data', blink_trace,...
    'data_unit', 'AU',...
    'timestamps',ts_video,...
    'timestamps_unit', 's',...
    'description', 'Blink trace extracted with Facemap algorithm form the pupil movies');

%% Face data
face = [face_area1; face_area2; face_area3];

% Face behavioral Timeseries
face_motion_data = types.core.TimeSeries( ...
    'data', face,...
    'data_unit', 'arbitrary',...
    'timestamps',ts_video,...
    'timestamps_unit', 's',...
    'description', '1st three face motion traces extracted with Facemap algorithm form the face movie');


%% Add_saturation Epochs (NON MANDATORY - CAN BE COMMENTED, some local path to change inside,...)

% In some recordings the LF and AP saturate. We here detect these event with a Schmitt trigger and
% timestamp them so they can be removed at the start of the analysis.
% this part is a bit dirty...but it works...


% binning window
bin_sz = 0.1; % in seconds
sr = 30000;
% Allen Brain Atlas regions to count the units, this data set is PFC
% focussed so we will only plot these ones:
Regions = {'MOs','ACA','PL','ILA','ORBm','ORBvl','ORBl'};
% Colors for the plot
HexColors = {'#99B898','#FECEA8','#FF847C','#6C5B7B','#355C7D','#355C7D','#355C7D'};

% get id and recording duration from meta file
id = strsplit(nwb.general_session_id,'_');

if strcmp(id{1},'PL026')
    Rec_num = num2str(6);
elseif strcmp(id{1},'PL035')
    Rec_num = num2str(3);
else
    Rec_num = num2str(0);
end

% Load 1LFP channel
lfp = nwb.processing.get('ecephys').nwbdatainterface.get('LFP').electricalseries.get('ElectricalSeries').data;
sr_lfp = nwb.processing.get('ecephys').nwbdatainterface.get('LFP').electricalseries.get('ElectricalSeries').starting_time_rate;
dims = size(lfp);

lfp = lfp(1,:);
t_lfp = linspace(1,round(size(lfp,2)/sr_lfp),size(lfp,2));

% Get rec duration
rec_dur = round(dims(2)/sr_lfp);
t = linspace(1,rec_dur,round(rec_dur/bin_sz));

% Load some binarize spike data from a local folder of bin and save it
% somewhere.
if exist(['/Users/pielem/Documents/MATLAB/neuropixelPFC/Matlab/analysis/Binned_FR/' identifier  '_binned_fr.mat'],'file')==2
    disp('Loading spikes')
    load(['/Users/pielem/Documents/MATLAB/neuropixelPFC/Matlab/analysis/Binned_FR/' identifier  '_binned_fr.mat'])
else
    disp(['Loading and Binning spikes in file ' identifier])
    % Get unit spiketimes
    % Load jagged arrays
    unit_times_data = nwb.units.spike_times.data;
    unit_times_idx = nwb.units.spike_times_index.data;
    unit_ids = nwb.units.id.data; % array of unit ids
    % Initialize times Map containers indexed by unit_ids
    unit_times = containers.Map('KeyType',class(unit_ids),'ValueType','any');
    last_idx = 0;
    for u = 1:length(unit_ids)
        unit_id = unit_ids(u);
        s_idx = last_idx + 1;
        e_idx = unit_times_idx(u);
        unit_times(unit_id) = unit_times_data(s_idx:e_idx);
        last_idx = e_idx;
    end

    % Bin Spikes
    spk_cnt = nan(numel(unit_ids),rec_dur/bin_sz);

    for k=1:numel(unit_ids)
        spike_ts =  unit_times(k-1).*sr; % -1 indexed in unit_times!!!
        [fr,t,fano,~] = firing_rate(spike_ts,sr,bin_sz,rec_dur);
        spk_cnt(k,:) = fr.*bin_sz;
    end

    save(['/Users/pielem/Documents/MATLAB/neuropixelPFC/Matlab/analysis/Binned_FR/' identifier  '_binned_fr.mat'],'spk_cnt')
end

% Detect saturation epoch on the LFP

% Schmitt trigger for the detection
Vh = double(max(lfp))-10;
Vl = double(max(lfp))-15;
[Sn]=Schmitt_Trigg(lfp,Vh,Vl);
Sn(Sn==0.5)=0;

Vh2 = double(max(lfp.*-1))-5;
Vl2 = double(max(lfp.*-1))-10;
[Sn2]=Schmitt_Trigg(lfp.*-1,Vh2,Vl2);
Sn2(Sn2==0.5)=0;

% Get in timestamps
dd = diff(Sn);
[~,in]=find(dd==1);
[~,out]=find(dd==-1);

dd2 = diff(Sn2);
[~,in2]=find(dd2==1);
[~,out2]=find(dd2==-1);


if strcmp(id{1},'PL034')
    out(1)=[];
elseif strcmp(id{1},'PL033')
    out2(1)=[];
elseif strcmp(id{1},'PL053')
    out(1)=[];
end

IN = sort([in in2]);
OUT = sort([out out2]);

% remove saturation event with duration <bin size
idx = [];
cnt = 1;
if numel(IN)>numel(OUT)
    IN(end) = [];
end

for i = 1:numel(IN)

    if (OUT(i)-IN(i))<bin_sz*sr_lfp
        idx(cnt) = i;
        cnt = cnt + 1;
    end
end
IN(idx) = [];
OUT(idx) = [];

%remove intervals with duration < bin size
idx2 = [];
cnt = 1;
for i = 1:numel(IN)-1
    if (IN(i+1)-OUT(i))<bin_sz*sr_lfp
        idx2(cnt) = i;
        cnt = cnt + 1;
    end
end
IN(idx2+1) = [];
OUT(idx2) = [];


% Write Saturation epochs in nwb
if isempty(IN)
    start = NaN;
    stop = NaN;
else
    start = IN./sr_lfp;
    stop = OUT./sr_lfp;
end

if strcmp(Mouse,'242820') % add values
    start = [0 start];
    stop = [1 start];
elseif strcmp(Mouse,'147463') % add values
    IN = [IN(1:2) IN(121:end)];
    OUT = [OUT(1) 6676500 OUT(121:end)];
    start = IN./sr_lfp;
    stop = OUT./sr_lfp;
end

% Write the LFP saturation data in the nwb file
saturationIN = types.core.TimeSeries(...
    'data', ones(1,numel(start)), ...
    'data_unit', 'none', ...
    'description', 'start times of the LFP saturation epoch', ...
    'timestamps', start, ...
    'timestamps_unit', 's');
saturationIN_name = 'LFP saturation start';
saturationIN_ref = types.untyped.ObjectView(['/analysis/', saturationIN_name]);
nwb.analysis.set(saturationIN_name, saturationIN);

saturationOUT = types.core.TimeSeries(...
    'data', zeros(1,numel(stop)), ...
    'data_unit', 'none', ...
    'description', 'stop times of the LFP saturation epoch', ...
    'timestamps', stop, ...
    'timestamps_unit', 's');
saturationOUT_name = 'LFP saturation stop';
saturationOUT_ref = types.untyped.ObjectView(['/analysis/', saturationOUT_name]);
nwb.analysis.set(saturationOUT_name, saturationOUT);


disp('Write LFP saturation times in nwb file')


% Detect saturation epoch on the spikes
pop_fr = mean(spk_cnt);

% Schmitt trigger
Vh = double(max(pop_fr.*-1))-0.05;
Vl = double(max(pop_fr.*-1))-0.1;
[Sn3]=Schmitt_Trigg(pop_fr.*-1,Vh,Vl);
%Sn3(Sn3==0.5)=0;

% Get in timestamps
dd3 = diff(Sn3);
[~,in3]=find(dd3==1);
[~,out3]=find(dd3==-1);

out3 = out3 + 1;


if numel(in3)>numel(out3)
    in3(end)=[];
end

if strcmp(id{1},'PL033')
else
    %remove saturation event with duration <2*bin size
    idx = [];
    cnt = 1;
    for i = 1:numel(in3)
        if (out3(i)-in3(i))<bin_sz*2
            idx(cnt) = i;
            cnt = cnt + 1;
        end
    end
    in3(idx) = [];
    out3(idx) = [];
end

% remove saturation events not within an LFP upward stauration
idx = [];
cnt = 1;
for i = 1:numel(in3)
    for j = 1 : numel(start)
        if (start(j)<t(in3(i)) && t(in3(i))<stop(j)) || (start(j)<t(out3(i)) && t(out3(i))<stop(j))
            idx(cnt) = i;
            cnt = cnt + 1;
        end
    end
end
in3 = in3(idx);
out3 = out3(idx);

% Write Saturation epochs in nwb
if isempty(in3)
    start2 = NaN;
    stop2 = NaN;
else
    start2 = t(in3);
    stop2 = t(out3);
end

% Write the spike saturation data in the nwb file
saturationIN2 = types.core.TimeSeries(...
    'data', ones(1,numel(start2)), ...
    'data_unit', 'none', ...
    'description', 'start times of the spike saturation epoch', ...
    'timestamps', start2, ...
    'timestamps_unit', 's');
saturationIN2_name = 'spike saturation start';
saturationIN2_ref = types.untyped.ObjectView(['/analysis/', saturationIN2_name]);
nwb.analysis.set(saturationIN2_name, saturationIN2);

saturationOUT2 = types.core.TimeSeries(...
    'data', zeros(1,numel(stop2)), ...
    'data_unit', 'none', ...
    'description', 'stop times of the spike saturation epoch', ...
    'timestamps', stop2, ...
    'timestamps_unit', 's');
saturationOUT2_name = 'spike saturation stop';
saturationOUT2_ref = types.untyped.ObjectView(['/analysis/', saturationOUT2_name]);
nwb.analysis.set(saturationOUT2_name, saturationOUT2);


disp('Write spike saturation times in nwb file')


% Reorganize raster by PFC Region
% Get regions
ede_regions = nwb.general_extracellular_ephys_electrodes.vectordata.get('location').data;
% Get unit main channel
main_ch = nwb.units.electrodes.data;

region_idx = zeros(1,numel(main_ch));
for R = 1:numel(Regions)
    region=char(Regions{R});
    % loop over units
    for i = 1 : numel(main_ch)
        if (strfind(char(ede_regions{main_ch(i)+1}),region)) > 0 %main_ch is 0 indexed !!
            region_idx(i) = R;
        end
    end
end

for R = 0:numel(Regions)
    eval(['im' num2str(R) ' = spk_cnt(region_idx==R,:);'])
end


% Plot the detected saturarion epochs 
fig1 = figure;
uu = unique(region_idx);
uu(uu==0)=[];
for R = 1:numel(uu)
    eval(['ax' num2str(R) ' = subplot(numel(uu)+2,1,R);'])
    eval(['imagesc(' '''XData''' ',t,' '''CData''' ',im' num2str(uu(R)) ');'])
    caxis([0, 5])
    ylabel(Regions{uu(R)})
    set(get(gca,'YLabel'),'Rotation',0)
    set(gca,'XTickLabel',[]);
    load(['/Users/pielem/Documents/MATLAB/neuropixelPFC/Matlab/Utilities/Colormaps/cmap_' Regions{uu(R)} '.mat']); % local color map here
    colormap(flipud(gray))
    eval(['colormap(cmap_' Regions{uu(R)} ')'])
    freezeColors %freeze this plot's colormap
end

eval(['ax' num2str(numel(Regions)+1) ' = subplot(numel(Regions)+2,1,numel(Regions)+1);'])
plot(t,mean(spk_cnt),'k')
hold on
for i = 1:numel(in3)
    line([t(in3(i)) t(in3(i))],[0 1],'LineWidth',1,'Color',[1 0 0])
    line([t(out3(i)) t(out3(i))],[0 1],'LineWidth',1,'Color',[0 0 1])
end
set(gca,'XTickLabel',[]);
title('population rate')

eval(['ax' num2str(numel(Regions)+2) ' = subplot(numel(Regions)+2,1,numel(Regions)+2);'])
plot(t_lfp,lfp,'b')
hold on
for i = 1:numel(IN)
    line([t_lfp(IN(i)) t_lfp(IN(i))],[min(lfp) max(lfp)],'LineWidth',1,'Color',[1 0 0])
    line([t_lfp(OUT(i)) t_lfp(OUT(i))],[min(lfp) max(lfp)],'LineWidth',1,'Color',[0 0 1])
end
title('LFP')
xlabel('Time(s)')

if numel(uu)==3
    linkaxes([ax1,ax2,ax3,ax8,ax9],'x')
elseif numel(uu)==4
    linkaxes([ax1,ax2,ax3,ax4,ax8,ax9],'x')
elseif numel(uu)==5
    linkaxes([ax1,ax2,ax3,ax4,ax5,ax8,ax9],'x')
elseif numel(uu)==6
    linkaxes([ax1,ax2,ax3,ax4,ax5,ax6,ax8,ax9],'x')
elseif numel(uu)==7
    linkaxes([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9],'x')
end
set(gcf,'units','points','position',[0,2400,1200,800])
sgtitle([id{1} ' Session ' id{2}])



%% Write NWB

% EXPORT TO DANDI !!! well localy first....
nwbExport(nwb, ['/Volumes/labs/pielem/DANDI_test/' identifier '.nwb']);
disp('NWB file saved in /Volumes/labs/pielem/DANDI_test/')




end % End of loop over mice