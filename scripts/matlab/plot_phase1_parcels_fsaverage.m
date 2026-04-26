% plot_phase1_parcels_fsaverage.m
%
% Visualize the v14 Tier-1 base parcels on fsaverage/lh.pial with the 7
% Phase-1 LH patients' electrodes overlaid on fsaverage pial positions.
%
% This is the fsaverage / Path A counterpart to plot_phase1_parcels.m.
% Blocker #36 closed 2026-04-16 late: Phase-1 spatial pipeline is strict
% fsaverage snap-to-pial. See docs/session_cras_fix_2026-04-16.md and
% data/atlas/fsaverage_parity/decision_memo.md for context.
%
% Differences from the cvs_avg35 version:
%   1. Target brain is fsaverage (cras=0, so no cras translation needed).
%   2. Atlas is read from the pre-baked multiframe .mgh (246 frames, one
%      per BNA parcel, 2D geodesic FWHM 3.5 mm). No volumetric trilinear.
%   3. Electrodes are read from data/fsaverage_coords/<pt>_fsaverage_pial.csv
%      which already contains fsaverage pial xyz -- no snap at render time.
%   4. Tier-1 parcels are the fsaverage-derived 15 (see token_spec.py).
%
% Vertex paint rule matches the electrode token_mask rule (blocker #3):
% a vertex is "in" Tier-1 iff its globally dominant parcel across all
% 246 BNA ROIs is one of the Tier-1 parcels. Non-Tier-1 argmax vertices
% stay gray even if their Tier-1 support is nonzero.
%
% Expects these to be set (see scripts/matlab/setup_ielvis.sh):
%   FREESURFER_HOME env var
%   SUBJECTS_DIR env var  (must contain fsaverage)
%   iELVis on the MATLAB path  (for tripatchDG)
%   $FREESURFER_HOME/matlab on the MATLAB path  (for read_surf, MRIread)

% ---- paths ----
here      = fileparts(mfilename('fullpath'));
proj_root = fileparts(fileparts(here));
csv_dir   = fullfile(proj_root, 'data', 'fsaverage_coords');
out_dir   = fullfile(proj_root, 'data', 'atlas', 'fsaverage_parity', 'figures_tier1');
bake_mgh  = fullfile(proj_root, 'data', 'atlas', 'fsaverage_bake_fast2', ...
                     'lh', 'lh.fsaverage.fwhm3.5.mgh');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

subjects_dir = getenv('SUBJECTS_DIR');
if isempty(subjects_dir)
    error('SUBJECTS_DIR is not set. Run scripts/matlab/setup_ielvis.sh first.');
end
avg_surf_dir = fullfile(subjects_dir, 'fsaverage', 'surf');

% ---- Tier-1 parcel definition (fsaverage-derived, 2026-04-16 late) ----
% Ordered by argmax_wins on the fsaverage cohort ranking; see
% src/speech_decoding/v14/token_spec.py for the canonical list.
parcels = { ...
    'A4hf',         53; ...   % PrG_L_6_1   wins=307  face/head M1
    'A1/2/3ulhf',  155; ...   % PoG_L_4_1   wins=246  face/head S1
    'A1/2/3tonIa', 157; ...   % PoG_L_4_2   wins=151  tongue S1
    'A4tl',         61; ...   % PrG_L_6_5   wins=110  tongue M1
    'IFJ',          17; ...   % MFG_L_7_2   wins= 89  inferior frontal junction
    'A44d',         29; ...   % IFG_L_6_1   wins= 65  Broca dorsal
    'A45c',         33; ...   % IFG_L_6_3   wins= 60  Broca caudal
    'A6cvl',        63; ...   % PrG_L_6_6   wins= 59  ventral premotor
    'A9/46v',       21; ...   % MFG_L_7_4   wins= 38  ventral 9/46
    'A40rd',       139; ...   % IPL_L_6_3   wins= 33  supramarginal rostrodorsal
    'IFS',          31; ...   % IFG_L_6_2   wins= 31  inferior frontal sulcus
    'A2',          159; ...   % PoG_L_4_3   wins= 26  area 2 (dorsal S1)
    'A40rv',       145; ...   % IPL_L_6_6   wins= 24  supramarginal rostroventral
    'A44v',         39; ...   % IFG_L_6_6   wins= 18  Broca ventral
    'A22r',         79  ...   % STG_L_6_6   wins= 17  rostral area 22
};
nParcels = size(parcels, 1);
parcel_names = parcels(:, 1);
tier1_bna    = cell2mat(parcels(:, 2));   % 1-indexed BNA indices

% iELVis palette, distinct from gray background
if exist('distinguishable_colors', 'file') == 2
    pColors = distinguishable_colors(nParcels, [0.75 0.75 0.75]);
else
    pColors = hsv(nParcels);
end

% ---- load fsaverage lh.pial ----
fprintf('loading fsaverage/lh.pial\n');
[lh_vert, lh_face] = read_surf(fullfile(avg_surf_dir, 'lh.pial'));
lh_face = lh_face + 1;
nVert = size(lh_vert, 1);
fprintf('  %d vertices, %d faces\n', nVert, size(lh_face, 1));

avgPial.vert = lh_vert;
avgPial.tri  = lh_face;

% ---- load baked BNA atlas on fsaverage (246 frames, FWHM 3.5 mm) ----
fprintf('loading fsaverage BNA bake (246 frames, FWHM 3.5 mm)\n');
pm = MRIread(bake_mgh);
% MRIread returns vol with shape [X Y Z nFrames]. Surface mgh has
% Y=Z=1 so [nVert 1 1 246]. Squeeze to [nVert 246].
all_pm_vert = squeeze(pm.vol);
if size(all_pm_vert, 1) ~= nVert
    error('bake vertex count %d != fsaverage lh.pial vertex count %d', ...
          size(all_pm_vert, 1), nVert);
end
if size(all_pm_vert, 2) ~= 246
    error('bake frame count %d != expected 246 BNA parcels', ...
          size(all_pm_vert, 2));
end
fprintf('  loaded [%d vertices x %d parcels]\n', ...
        size(all_pm_vert, 1), size(all_pm_vert, 2));

% ---- full-atlas argmax per vertex ----
[full_max, full_argmax] = max(all_pm_vert, [], 2);   % nVert x 1, 1-indexed
full_argmax = full_argmax(:);
full_max    = full_max(:);

% map each vertex to its Tier-1 slot (1..nParcels) or 0 if outside Tier-1
assigned = zeros(nVert, 1);
for p = 1:nParcels
    assigned(full_argmax == tier1_bna(p)) = p;
end
assigned(full_max == 0) = 0;  % medial wall / unsupported -> gray

fprintf('per-parcel vertex counts:\n');
for p = 1:nParcels
    n_assigned = nnz(assigned == p);
    fprintf('  %-14s  %6d vertices  (BNA %d)\n', ...
            parcel_names{p}, n_assigned, tier1_bna(p));
end
fprintf('  %-14s  %6d vertices (gray)\n', '(unassigned)', nnz(assigned == 0));

% ---- per-vertex color ----
base_color = [0.75 0.75 0.75];
blend_max  = 0.9;
vcol = repmat(base_color, nVert, 1);
for p = 1:nParcels
    mask   = (assigned == p);
    n_mask = nnz(mask);
    if n_mask == 0; continue; end
    pcol = (1 - blend_max) * base_color + blend_max * pColors(p, :);
    vcol(mask, :) = repmat(pcol, n_mask, 1);
end

% ---- patient config ----
patients = {'S14', 'S16', 'S23', 'S26', 'S33', 'S39', 'S62'};
elecColor = [1 1 1];

% ---- pooled all-patient lateral view ----
fprintf('plotting pooled all-patient lateral view\n');
hFig = figure('Position', [100 100 1000 700], 'Color', 'w', ...
              'Name', 'fsaverage: all patients + Tier-1 parcels');
tripatchDG(avgPial, hFig, vcol, 'EdgeColor', 'none');
shading interp; lighting gouraud; material dull; axis off; axis vis3d; hold on;
alpha(1.0);
light('Position', [-1 0 0]);
view(270, 0);
rotate3d on;

total_n = 0;
for k = 1:numel(patients)
    pt      = patients{k};
    csvFile = fullfile(csv_dir, sprintf('%s_fsaverage_pial.csv', pt));
    if ~exist(csvFile, 'file')
        warning('missing %s -- skipping', csvFile);
        continue;
    end
    T = readtable(csvFile, 'TextType', 'string');
    % filter to left hemisphere (Phase-1 is LH-only)
    lh_mask = strcmp(T.hemisphere, "L");
    T = T(lh_mask, :);
    scatter3(T.x, T.y, T.z, 14, ...
             'MarkerFaceColor', elecColor, ...
             'MarkerEdgeColor', 'k', 'LineWidth', 0.2);
    total_n = total_n + height(T);
end
title(sprintf(['Phase-1 LH electrodes on fsaverage Tier-1 parcels  ' ...
               '(N=%d, pooled; bake=FWHM 3.5 mm)'], total_n), ...
      'Color', 'k', 'Interpreter', 'none');
png = fullfile(out_dir, 'all_patients_parcels_tier1_fsaverage_lateral.png');
print(hFig, png, '-dpng', '-r150');
fprintf('wrote %s\n', png);

% ---- pooled medial view ----
fprintf('plotting pooled all-patient medial view\n');
hFig = figure('Position', [100 100 1000 700], 'Color', 'w', ...
              'Name', 'fsaverage: all patients + Tier-1 parcels (medial)');
tripatchDG(avgPial, hFig, vcol, 'EdgeColor', 'none');
shading interp; lighting gouraud; material dull; axis off; axis vis3d; hold on;
alpha(0.75);
light('Position', [1 0 0]);
view(90, 0);
for k = 1:numel(patients)
    pt      = patients{k};
    csvFile = fullfile(csv_dir, sprintf('%s_fsaverage_pial.csv', pt));
    if ~exist(csvFile, 'file'); continue; end
    T = readtable(csvFile, 'TextType', 'string');
    T = T(strcmp(T.hemisphere, "L"), :);
    scatter3(T.x, T.y, T.z, 14, ...
             'MarkerFaceColor', elecColor, ...
             'MarkerEdgeColor', 'k', 'LineWidth', 0.2);
end
title('Phase-1 LH electrodes on fsaverage Tier-1 parcels (medial)', ...
      'Color', 'k', 'Interpreter', 'none');
png = fullfile(out_dir, 'all_patients_parcels_tier1_fsaverage_medial.png');
print(hFig, png, '-dpng', '-r150');
fprintf('wrote %s\n', png);

% ---- parcel legend ----
fprintf('writing parcel legend\n');
hFig = figure('Position', [100 100 450 520], 'Color', 'w', ...
              'Name', 'Tier-1 parcel legend (fsaverage)');
axis off; hold on;
for p = 1:nParcels
    yy = nParcels - p + 1;
    patch([0 1 1 0], [yy-0.4 yy-0.4 yy+0.4 yy+0.4], pColors(p,:), 'EdgeColor', 'k');
    text(1.15, yy, sprintf('%s  (%d)', parcel_names{p}, tier1_bna(p)), ...
         'FontSize', 11, 'FontName', 'Menlo', 'VerticalAlignment', 'middle');
end
xlim([0 6]); ylim([0 nParcels+1]);
title('v14 Tier-1 parcels on fsaverage (15 parcels, argmax_wins >= 10)', ...
      'FontSize', 12, 'Interpreter', 'none');
png = fullfile(out_dir, 'parcel_legend_tier1_fsaverage.png');
print(hFig, png, '-dpng', '-r150');
fprintf('wrote %s\n', png);

fprintf('\ndone. outputs in %s\n', out_dir);
