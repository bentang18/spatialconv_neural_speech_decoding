% plot_phase1_parcels.m
%
% Visualize the v14 Tier-1 base parcels on cvs_avg35_inMNI152/lh.pial
% with RAW Phase-1 LH electrodes overlaid (no pial snap).
%
% Loads Brainnetome PM (4D, 246 ROIs), samples ALL 246 parcels at each
% pial vertex, and takes the full-atlas argmax. A vertex is painted iff
% its argmax lands inside the Tier-1 set (same rule the electrode
% `token_mask` uses, dual to blocker #4's Tier-1 selection). Vertices
% whose global argmax is a non-Tier-1 parcel (e.g. leakage sinks like
% A2/A4t, or out-of-cohort territory like IPL/SMG) stay gray even if
% their Tier-1 support is nonzero. This replaces the earlier thr=10
% vertex paint rule, which produced false "gaps" where Tier-1 PM simply
% faded below 10% while electrodes in the pocket were still correctly
% argmax'd into Tier-1 at 40-70% support.
%
% 2026-04-14 spatial pipeline (final Phase-1 decision):
%   1. sub2AvgBrainClinical produces raw MNI coords on cvs_avg35's dural
%      envelope (pial-outer-smoothed). See scripts/preprocess_mni_coordinates.py.
%   2. BNA PM is dilated outward by d_max=8 mm via nearest-neighbor
%      propagation (scripts/dilate_pm.py, scipy.ndimage.distance_transform_edt).
%      Dilation covers the 2-5 mm cvs_avg35-pial-vs-BNA-ribbon mismatch
%      AND the 2-4 mm dural-to-pial offset in a single step -- no electrode
%      snap needed. Raw MNI coords give 1280/1280 full coverage at
%      max_parcel_support >= 10% for every Phase-1 LH patient.
%   3. Gaussian PSF smoothing sigma=1.5 mm (Trumpis 2020 uECoG PSF) via
%      scipy.ndimage.gaussian_filter.
%   4. Trilinear sampling at raw MNI positions.
%
% Parcel set (15, data-driven from raw + 8mm dilation ranking under the
% argmax-centric rule `argmax_wins >= 10`):
% A parcel is Tier-1 iff at least 10 cohort electrodes have it as their
% globally dominant parcel across all 246 BNA ROIs. This directly measures
% "how many electrodes anatomically live here" rather than using eff_N as
% a proxy. Robust cutoff: the [5, 9] argmax band is empty on current data.
% TE1.0/1.2 (primary auditory, 73) made it under this rule after failing
% the earlier `eff_N >= 10` clause on a technicality (eff_N=7.14 but
% argmax=12). A38l (77) is still flagged for visual verification: its
% support is S26-dominated with max_pm=41.5%, so if this render shows
% S26's grid NOT reaching the lateral anterior temporal pole, A38l is
% dilation smearing and should be dropped (N_tok -> 14). Aggregations
% (STGa/STGpp/INSa/MFG) remain dropped.
%
% Electrodes are read from <pt>_MNI152.csv directly (no snap). Flip
% PM_FNAME below to 'BNA_PM_dilated_5mm.nii.gz' for the safer variant
% or 'BNA_PM_4D.nii.gz' for the undilated original.

% ---- paths ----
here      = fileparts(mfilename('fullpath'));
proj_root = fileparts(fileparts(here));
csv_dir   = fullfile(proj_root, 'data', 'mni_coords');
out_dir   = fullfile(csv_dir, 'viz_ielvis');
PM_FNAME  = 'BNA_PM_dilated_8mm.nii.gz';   % flip to 'BNA_PM_4D.nii.gz' or 'BNA_PM_dilated_5mm.nii.gz'
pm_path   = fullfile(proj_root, 'data', 'atlas', PM_FNAME);
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

subjects_dir = getenv('SUBJECTS_DIR');
if isempty(subjects_dir)
    error('SUBJECTS_DIR is not set. Run scripts/matlab/setup_ielvis.sh first.');
end
avg_surf_dir = fullfile(subjects_dir, 'cvs_avg35_inMNI152', 'surf');
avg_mri_dir  = fullfile(subjects_dir, 'cvs_avg35_inMNI152', 'mri');

% ---- parcel definition (LH Brainnetome indices, Tier-1 data-driven set) ----
% Ordered by effective_N from raw + 8mm dilation ranking (high -> low).
% All six LH patients now contribute to most top frontal parcels because
% S26's grid -- previously in a BNA pocket at the lateral gyral crowns --
% lands in-parcel after the PM dilation.
parcels = { ...
    'A45c',          33; ...   % IFG_L_6_3   argmx=223  eff_N=202.43  pts=6
    'A45r',          35; ...   % IFG_L_6_4   argmx=199  eff_N=179.17  pts=6
    'IFS',           31; ...   % IFG_L_6_2   argmx=181  eff_N=188.61  pts=6
    'A4hf',          53; ...   % PrG_L_6_1   argmx=168  eff_N=121.19  pts=5
    'A44v',          39; ...   % IFG_L_6_6   argmx=123  eff_N=112.55  pts=6
    'A9/46v',        21; ...   % MFG_L_7_4   argmx=100  eff_N= 77.02  pts=4
    'A12/47l',       51; ...   % OrG_L_6_6   argmx= 48  eff_N= 32.08  pts=2
    'A4tl',          61; ...   % PrG_L_6_5   argmx= 44  eff_N= 52.15  pts=5
    'A38l',          77; ...   % STG_L_6_5   argmx= 40  eff_N= 15.60  pts=3  (S26-dominated, verify visually)
    'A44d',          29; ...   % IFG_L_6_1   argmx= 34  eff_N= 48.13  pts=6
    'A1/2/3tonIa',  157; ...   % PoG_L_4_2   argmx= 29  eff_N= 26.93  pts=2
    'A1/2/3ulhf',   155; ...   % PoG_L_4_1   argmx= 28  eff_N= 30.10  pts=2
    'IFJ',           17; ...   % MFG_L_7_2   argmx= 27  eff_N= 23.65  pts=4
    'A6cvl',         63; ...   % PrG_L_6_6   argmx= 24  eff_N= 31.94  pts=5
    'TE1.0/1.2',     73  ...   % STG_L_6_3   argmx= 12  eff_N=  7.14  pts=3  primary auditory (Heschl's)
};
nParcels = size(parcels, 1);
parcel_names = parcels(:, 1);

% categorical palette, distinct from gray background
% iELVis ships distinguishable_colors.m; fall back to hsv if unavailable.
if exist('distinguishable_colors', 'file') == 2
    pColors = distinguishable_colors(nParcels, [0.75 0.75 0.75]);
else
    pColors = hsv(nParcels);
end

% ---- load lh.pial ----
fprintf('loading lh.pial\n');
[lh_vert, lh_face] = read_surf(fullfile(avg_surf_dir, 'lh.pial'));
lh_face = lh_face + 1;
nVert = size(lh_vert, 1);
fprintf('  %d vertices, %d faces\n', nVert, size(lh_face, 1));

avgPial.vert = lh_vert;
avgPial.tri  = lh_face;

% ---- load PM_4D ----
fprintf('loading Brainnetome PM (4D, 246 ROIs)\n');
pm = MRIread(pm_path);
fprintf('  shape = %s\n', mat2str(size(pm.vol)));

% ---- cras offset (tkrRAS -> scanner RAS) ----
fprintf('computing cras from orig.mgz\n');
orig = MRIread(fullfile(avg_mri_dir, 'orig.mgz'));
sz = size(orig.vol);
center_vox = [sz(2); sz(1); sz(3)] / 2;
cras_h = orig.vox2ras0 * [center_vox; 1];
cras_offset = cras_h(1:3);
fprintf('  cras = [%.3f %.3f %.3f]\n', cras_offset);

% flip if PM overlay appears offset after first run
use_cras = true;
if use_cras
    vert_scanner = lh_vert + cras_offset';
else
    vert_scanner = lh_vert;
end

% ---- convert vertex RAS -> PM voxel indices ----
ras4 = [vert_scanner, ones(nVert, 1)]';
vox  = pm.vox2ras0 \ ras4;                 % 4 x nVert, 0-indexed
c = vox(1, :) + 1;                         % 1-indexed
r = vox(2, :) + 1;
s = vox(3, :) + 1;

% ---- sample PM for ALL 246 BNA parcels at each vertex, full argmax ----
% Matches the electrode-level token_mask rule: a vertex is "in" Tier-1
% iff its globally dominant parcel across all 246 BNA ROIs is one of
% the Tier-1 parcels. No PM threshold. Vertices whose full-atlas argmax
% lands in a non-Tier-1 parcel stay gray regardless of Tier-1 support.
fprintf('sampling all 246 BNA parcels at %d vertices\n', nVert);
all_pm_vert = zeros(246, nVert);
for pi = 1:246
    all_pm_vert(pi, :) = interp3(pm.vol(:, :, :, pi), c, r, s, 'linear', 0);
end

[full_max, full_argmax] = max(all_pm_vert, [], 1);   % 1 x nVert, 1-indexed
full_argmax = full_argmax(:);
full_max    = full_max(:);

% Map each vertex to its Tier-1 slot (1..nParcels) or 0 if outside Tier-1
tier1_bna = cell2mat(parcels(:, 2));   % nParcels x 1, 1-indexed BNA
assigned  = zeros(nVert, 1);
for p = 1:nParcels
    assigned(full_argmax == tier1_bna(p)) = p;
end
% Medial wall / all-zero vertices stay gray even if accidentally argmax=1
assigned(full_max == 0) = 0;

fprintf('per-parcel vertex counts:\n');
for p = 1:nParcels
    n_assigned = nnz(assigned == p);
    fprintf('  %-14s  %6d vertices  (%d ROIs)\n', parcel_names{p}, n_assigned, numel(parcels{p,2}));
end
fprintf('  %-14s  %6d vertices (gray)\n', '(unassigned)', nnz(assigned == 0));

% ---- per-vertex color ----
base_color = [0.75 0.75 0.75];
blend_max  = 0.9;
vcol = repmat(base_color, nVert, 1);
for p = 1:nParcels
    mask = (assigned == p);
    n_mask = nnz(mask);
    if n_mask == 0; continue; end
    pcol = (1 - blend_max) * base_color + blend_max * pColors(p, :);  % 1x3
    vcol(mask, :) = repmat(pcol, n_mask, 1);
end

% ---- patient config ----
patients = {'S14', 'S16', 'S23', 'S26', 'S33', 'S39', 'S62'};
elecColor = [1 1 1];   % all electrodes white

% ---- pooled all-patient lateral view ----
fprintf('plotting pooled all-patient view\n');
hFig = figure('Position', [100 100 1000 700], 'Color', 'w', 'Name', 'all patients + parcels');
tripatchDG(avgPial, hFig, vcol, 'EdgeColor', 'none');
shading interp; lighting gouraud; material dull; axis off; axis vis3d; hold on;
alpha(0.75);
light('Position', [-1 0 0]);
view(270, 0);
rotate3d on;

total_n = 0;
for k = 1:numel(patients)
    pt      = patients{k};
    csvFile = fullfile(csv_dir, sprintf('%s_MNI152.csv', pt));
    if ~exist(csvFile, 'file')
        warning('missing %s -- skipping', csvFile);
        continue;
    end
    T = readtable(csvFile, 'TextType', 'string');
    scatter3(T.x, T.y, T.z, 14, ...
             'MarkerFaceColor', elecColor, ...
             'MarkerEdgeColor', 'k', 'LineWidth', 0.2);
    total_n = total_n + height(T);
end
pm_stem = regexprep(PM_FNAME, '\.nii(\.gz)?$', '');
title(sprintf('Phase-1 LH raw electrodes on Tier-1 parcels  (N=%d, pooled, PM=%s)', ...
      total_n, pm_stem), 'Color', 'k', 'Interpreter', 'none');
png = fullfile(out_dir, sprintf('all_patients_parcels_tier1_raw_lateral_%s.png', pm_stem));
print(hFig, png, '-dpng', '-r150');
fprintf('wrote %s\n', png);

% ---- parcel legend (separate figure: each parcel as a colored patch) ----
fprintf('writing parcel legend\n');
hFig = figure('Position', [100 100 450 520], 'Color', 'w', 'Name', 'parcel legend');
axis off; hold on;
for p = 1:nParcels
    yy = nParcels - p + 1;
    patch([0 1 1 0], [yy-0.4 yy-0.4 yy+0.4 yy+0.4], pColors(p,:), 'EdgeColor', 'k');
    text(1.15, yy, parcel_names{p}, ...
         'FontSize', 11, 'FontName', 'Menlo', 'VerticalAlignment', 'middle');
end
xlim([0 5]); ylim([0 nParcels+1]);
title('v14 Tier-1 parcels (15 parcels, rule: argmax_wins >= 10)', 'FontSize', 12);
png = fullfile(out_dir, sprintf('parcel_legend_tier1_%s.png', pm_stem));
print(hFig, png, '-dpng', '-r150');
fprintf('wrote %s\n', png);

fprintf('\ndone. outputs in %s\n', out_dir);
