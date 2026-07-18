"""Pyodide adaptation of the interactive linear registration workflow.

Runs entirely in the browser via Pyodide (no server required).
Loaded and executed by pyodide_worker.ts as the user script.

All behaviour is identical to example_linear_registration.py except:
  - No argparse / CLI: demo data is used automatically.
  - threading.Timer-based debounce replaced with JS setTimeout.
  - File writes (np.savetxt, json.dump) trigger browser downloads instead.
  - No blocking input() or webbrowser.open_new() calls.
"""

import json
import logging
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pprint import pprint
from time import ctime, time

import js
import neuroglancer
import numpy as np
import scipy.ndimage
from pyodide.ffi import create_proxy

DEBUG = False  # Print debug info during execution
MESSAGE_DURATION = 4  # How long to show help messages in seconds
NUM_DEMO_DIMS = 3  # Only used if no data given, can be 2D or 3D
NUM_NEAREST_POINTS = 4  # Number of nearest points to use in local estimation
AFFINE_NUM_DECIMALS = 6  # Number of decimals to round affine matrix to

# We make a copy of all the physical dimensions, but to avoid
# expecting a copy of dimensions like t, or time, they are listed here
# channel dimensions are already handled separately and don't need to be listed here
NON_PHYSICAL_DIM_NAMES = ["t", "time"]

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Browser download helper
# ---------------------------------------------------------------------------


def _browser_download(
    filename: str, content: str, mime_type: str = "application/octet-stream"
):
    """Trigger a file download via the main thread (workers lack document)."""
    from pyodide.ffi import to_js  # type: ignore[import]

    js.self.postMessage(
        to_js(
            {
                "type": "download",
                "filename": filename,
                "content": content,
                "mimeType": mime_type,
            },
            dict_converter=js.Object.fromEntries,
        )
    )


# ---------------------------------------------------------------------------
# Debounce using JS setTimeout instead of threading.Timer
# ---------------------------------------------------------------------------


def debounce(wait: float):
    """Wrap function in debounce using JS setTimeout."""

    def decorator(fn):
        timer_id = [None]
        proxy_ref = [None]

        def debounced(*args, **kwargs):
            if timer_id[0] is not None:
                js.clearTimeout(timer_id[0])
                if proxy_ref[0] is not None:
                    proxy_ref[0].destroy()
            proxy_ref[0] = create_proxy(lambda: fn(*args, **kwargs))
            timer_id[0] = js.setTimeout(proxy_ref[0], int(wait * 1000))

        return debounced

    return decorator


# ---------------------------------------------------------------------------
# Transform estimation (unchanged from example_linear_registration.py)
# ---------------------------------------------------------------------------


def estimate_transform(
    fixed_points: np.ndarray, moving_points: np.ndarray, force_non_affine=False
):
    assert fixed_points.shape == moving_points.shape
    N, D = fixed_points.shape

    if N == 1:
        return translation_fit(fixed_points, moving_points)
    elif N == 2:
        return rigid_or_similarity_fit(fixed_points, moving_points, rigid=True)
    elif N == 3 and D == 2:
        return affine_fit(fixed_points, moving_points)
    elif (N == 3 and D > 2) or force_non_affine:
        return rigid_or_similarity_fit(fixed_points, moving_points, rigid=False)
    return affine_fit(fixed_points, moving_points)


def translation_fit(fixed_points: np.ndarray, moving_points: np.ndarray):
    N, D = fixed_points.shape
    estimated_translation = np.mean(fixed_points - moving_points, axis=0)
    affine = np.zeros((D, D + 1))
    affine[:, :D] = np.eye(D)
    affine[:, -1] = estimated_translation
    affine = np.round(affine, decimals=AFFINE_NUM_DECIMALS)
    return affine


def rigid_or_similarity_fit(
    fixed_points: np.ndarray, moving_points: np.ndarray, rigid=True
):
    N, D = fixed_points.shape
    mu_q = moving_points.mean(axis=0)
    mu_p = fixed_points.mean(axis=0)
    Q = moving_points - mu_q
    P = fixed_points - mu_p
    H = (P.T @ Q) / N
    U, Sigma, Vt = np.linalg.svd(H)
    d = np.ones(D)
    if np.linalg.det(U @ Vt) < 0:
        d[-1] = -1.0
    R = U @ np.diag(d) @ Vt
    if rigid:
        s = 1.0
    else:
        var_x = (Q**2).sum() / N
        s = (Sigma * d).sum() / var_x
    t = mu_p - s * (R @ mu_q)
    T = np.zeros((D, D + 1))
    T[:D, :D] = s * R
    T[:, -1] = t
    affine = np.round(T, decimals=AFFINE_NUM_DECIMALS)
    return affine


def affine_fit(fixed_points: np.ndarray, moving_points: np.ndarray):
    N, D = fixed_points.shape
    Q = np.zeros(((D * N), D * (D + 1)))
    for i in range(N):
        for j in range(D):
            start_index = j * D
            end_index = (j + 1) * D
            Q[D * i + j, start_index:end_index] = moving_points[i]
            Q[D * i + j, D * D + j] = 1
    P = fixed_points.flatten()
    tvec, res, rank, sd = np.linalg.lstsq(Q, P)
    if rank < D * (D + 1):
        return rigid_or_similarity_fit(fixed_points, moving_points, rigid=False)
    affine = np.zeros((D, D + 1))
    for i in range(D):
        start_index = i * D
        end_index = start_index + D
        affine[i, :D] = tvec[start_index:end_index]
        affine[i, -1] = tvec[D * D + i]
    affine = np.round(affine, decimals=AFFINE_NUM_DECIMALS)
    return affine


def transform_points(affine: np.ndarray, points: np.ndarray):
    transformed = np.zeros_like(points)
    padded = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
    for i in range(len(points)):
        transformed[i] = affine @ padded[i]
    return transformed


# ---------------------------------------------------------------------------
# Demo data helpers (unchanged)
# ---------------------------------------------------------------------------


def _create_demo_data(size: int | tuple = 60, radius: float = 20):
    data_size = (size,) * NUM_DEMO_DIMS if isinstance(size, int) else size
    data = np.zeros(data_size, dtype=np.uint8)
    if NUM_DEMO_DIMS == 2:
        yy, xx = np.indices(data.shape)
        center = np.array(data.shape) / 2
        circle_mask = (xx - center[1]) ** 2 + (yy - center[0]) ** 2 < radius**2
        data[circle_mask] = 255
        return data
    zz, yy, xx = np.indices(data.shape)
    center = np.array(data.shape) / 2
    sphere_mask = (xx - center[2]) ** 2 + (yy - center[1]) ** 2 + (
        zz - center[0]
    ) ** 2 < radius**2
    data[sphere_mask] = 255
    return data


def _create_demo_fixed_image():
    return neuroglancer.ImageLayer(
        source=[
            neuroglancer.LayerDataSource(neuroglancer.LocalVolume(_create_demo_data()))
        ]
    )


def _create_demo_moving_image():
    if NUM_DEMO_DIMS == 2:
        desired_output_matrix_homogenous = [
            [0.8, 0, 0],
            [0, 0.2, 0],
            [0, 0, 1],
        ]
    else:
        desired_output_matrix_homogenous = [
            [0.8, 0, 0, 0],
            [0, 0.2, 0, 0],
            [0, 0, 0.9, 0],
            [0, 0, 0, 1],
        ]
    inverse_matrix = np.linalg.inv(desired_output_matrix_homogenous)
    transformed = scipy.ndimage.affine_transform(
        _create_demo_data(),
        matrix=inverse_matrix,
    )
    print("Target demo affine, can be compared to estimated", inverse_matrix)
    return neuroglancer.ImageLayer(
        source=[neuroglancer.LayerDataSource(neuroglancer.LocalVolume(transformed))]
    )


# ---------------------------------------------------------------------------
# Coord space helpers (unchanged)
# ---------------------------------------------------------------------------


def filter_local_dims(
    space: neuroglancer.CoordinateSpace,
) -> neuroglancer.CoordinateSpace:
    """Return a new CoordinateSpace with local channel dims (c^, c', c#, etc.) removed."""
    indices = [i for i, n in enumerate(space.names) if not n.endswith(("'", "^", "#"))]
    return neuroglancer.CoordinateSpace(
        names=[space.names[i] for i in indices],
        units=[space.units[i] for i in indices],
        scales=np.array([space.scales[i] for i in indices]),
    )


def copy_coord_space(space: neuroglancer.CoordinateSpace, name_suffix):
    def change_name(n):
        if n.endswith(("'", "^", "#")):
            return n
        return n + name_suffix

    return neuroglancer.CoordinateSpace(
        names=[change_name(n) for n in space.names],
        units=space.units,
        scales=space.scales,  # type: ignore
    )


def create_coord_space_matching_global_dims(
    viewer_dims: neuroglancer.CoordinateSpace, indices=None
):
    names = viewer_dims.names
    units = viewer_dims.units
    scales = viewer_dims.scales
    if indices is not None:
        return neuroglancer.CoordinateSpace(
            names=[names[i] for i in indices],
            units=[units[i] for i in indices],
            scales=np.array([scales[i] for i in indices]),
        )
    return neuroglancer.CoordinateSpace(names=names, units=units, scales=scales)


# ---------------------------------------------------------------------------
# Pipeline state enums (unchanged)
# ---------------------------------------------------------------------------


class PipelineState(Enum):
    NOT_READY = 0
    COORDS_READY = 1
    READY = 2
    ERROR = 3


class PointFilter(Enum):
    NONE = 0
    NEAREST = 1


# ---------------------------------------------------------------------------
# Main workflow class
# ---------------------------------------------------------------------------


class LinearRegistrationWorkflow:
    def __init__(
        self,
        starting_ng_state=None,
        annotations_name="annotation",
        unlink_scales=False,
        output_name="affine.txt",
    ):
        self.annotations_name = annotations_name
        self.pipeline_state = PipelineState.NOT_READY
        self.unlink_scales = unlink_scales
        self.output_name = output_name

        self.stored_points = ([], [], False)
        self.stored_map_moving_name_to_data_coords = {}
        self.stored_map_moving_name_to_viewer_coords = {}
        self.affine = None
        self.viewer = neuroglancer.Viewer()
        self.viewer.shared_state.add_changed_callback(
            lambda: self.viewer.defer_callback(self.update)
        )

        self._last_updated_print_time = -1
        self._status_timers = {}
        self._current_moving_layer_idx = 0
        self._cached_moving_layer_names = []
        self._force_non_affine = False
        self._annotation_filter_method = PointFilter.NONE

        linear_reg_pipeline_info = None
        if starting_ng_state is None:
            self._add_demo_data_to_viewer()
        else:
            linear_reg_pipeline_info = starting_ng_state.to_json().get(
                "linear_reg_pipeline_info", None
            )
            self.viewer.set_state(starting_ng_state)

        self._setup_viewer_actions()
        self._show_help_message()

        if linear_reg_pipeline_info is not None:
            self._restore_coord_maps(linear_reg_pipeline_info)
            self.pipeline_state = PipelineState.READY

        if self.pipeline_state == PipelineState.NOT_READY:
            self.setup_initial_two_panel_layout()

    def update(self):
        current_time = time()
        if current_time - self._last_updated_print_time > 5:
            print(f"Viewer states are successfully syncing at {ctime()}")
            self._last_updated_print_time = current_time
        if self.pipeline_state == PipelineState.COORDS_READY:
            self.setup_registration_point_layer()
        elif self.pipeline_state == PipelineState.ERROR:
            return
        elif self.pipeline_state == PipelineState.READY:
            self.update_affine()
        self._clear_status_messages()

    def _reset(self):
        self._cached_moving_layer_names = []
        self._current_moving_layer_idx = 0
        self.stored_map_moving_name_to_data_coords = {}
        self.stored_map_moving_name_to_viewer_coords = {}

    def setup_initial_two_panel_layout(self):
        with self.viewer.txn() as s:
            all_layer_names = [layer.name for layer in s.layers]
            if len(all_layer_names) >= 2:
                last_layer_name = all_layer_names[-1]
                group1_names = all_layer_names[:-1]
                group2_names = [last_layer_name]
            else:
                group1_names = all_layer_names
                group2_names = all_layer_names
            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(layers=group1_names, layout="xy-3d"),
                    neuroglancer.LayerGroupViewer(layers=group2_names, layout="xy-3d"),
                ]
            )
            s.layout.children[1].crossSectionOrientation.link = "unlinked"
            s.layout.children[1].projectionOrientation.link = "unlinked"

            if self.unlink_scales:
                s.layout.children[1].crossSectionScale.link = "unlinked"
                s.layout.children[1].projectionScale.link = "unlinked"

    def setup_viewer_after_user_ready(self):
        self._copy_moving_layers_to_left_panel()
        self.setup_second_coord_space()

    def setup_second_coord_space(self):
        layer_name = self._cached_moving_layer_names[self._current_moving_layer_idx]
        info_future = self.viewer.volume_info(layer_name)
        info_future.add_done_callback(lambda f: self._update_coord_space_info_cache(f))

    def setup_registration_point_layer(self):
        with self.viewer.txn() as s:
            if (
                self.pipeline_state == PipelineState.ERROR
                or not self.has_two_coord_spaces(s)
            ):
                self._show_help_message()
                return

            if s.layers.index(self.annotations_name) == -1:
                s.layers[self.annotations_name] = neuroglancer.LocalAnnotationLayer(
                    dimensions=create_coord_space_matching_global_dims(s.dimensions)
                )
            self._ignore_non_display_dims(s)

            s.layers[self.annotations_name].tool = "annotatePoint"
            s.selected_layer.layer = self.annotations_name
            s.selected_layer.visible = True
            s.layout.children[0].layers.append(self.annotations_name)
            s.layout.children[1].layers.append(self.annotations_name)
            self.setup_panel_display_dims(s)
            self.pipeline_state = PipelineState.READY
            self._show_help_message()

    def setup_panel_display_dims(self, s: neuroglancer.ViewerState):
        fixed_dims, moving_dims = self.get_fixed_and_moving_dims(s)
        s.layout.children[1].displayDimensions.link = "unlinked"
        s.layout.children[1].displayDimensions.value = moving_dims[:3]
        s.layout.children[0].displayDimensions.link = "unlinked"
        s.layout.children[0].displayDimensions.value = fixed_dims[:3]

    def _update_coord_space_info_cache(self, info_future):
        self.moving_name = self._cached_moving_layer_names[
            self._current_moving_layer_idx
        ]
        try:
            result = info_future.result()
            self.stored_map_moving_name_to_data_coords[self.moving_name] = (
                filter_local_dims(result.dimensions)
            )
        except Exception as e:
            # volume_info fails for external (non-python://) data sources because
            # findMatchingSource can't match scales for layers with a unit conversion
            # in modelToRenderLayerTransform (e.g. 8nm EM data in a 1μm viewer).
            # Use only the fixed (non-primed) spatial dims from the viewer as the
            # moving data's native coordinate space. If a starting URL with primed
            # dims is provided, s.dimensions includes both fixed and moving dims;
            # using all of them would cause copy_coord_space to produce x22,y22,z22
            # instead of x2,y2,z2.
            print(
                f"volume_info failed for {self.moving_name} ({e}); "
                "falling back to fixed spatial dimensions"
            )
            with self.viewer.txn() as s:
                fixed_dims, _ = self.get_fixed_and_moving_dims(s)
                all_dims = s.dimensions
                indices = [i for i, n in enumerate(all_dims.names) if n in fixed_dims]
                if indices:
                    data_coords = neuroglancer.CoordinateSpace(
                        names=[all_dims.names[i] for i in indices],
                        units=[all_dims.units[i] for i in indices],
                        scales=np.array([all_dims.scales[i] for i in indices]),
                    )
                else:
                    data_coords = all_dims
                self.stored_map_moving_name_to_data_coords[self.moving_name] = (
                    data_coords
                )

        self._current_moving_layer_idx += 1
        if self._current_moving_layer_idx < len(self._cached_moving_layer_names):
            self.setup_second_coord_space()
        else:
            return self._create_second_coord_space()

    def _create_second_coord_space(self):
        if self.pipeline_state == PipelineState.ERROR:
            return self.pipeline_state
        self.pipeline_state = PipelineState.COORDS_READY
        with self.viewer.txn() as s:
            for layer_name in self._cached_moving_layer_names:
                output_dims = self.stored_map_moving_name_to_data_coords.get(
                    layer_name, None
                )
                if output_dims is None:
                    print(
                        f"ERROR: could not get output dims for a moving layer {layer_name}"
                    )
                    self.pipeline_state = PipelineState.ERROR
                    continue
                self.stored_map_moving_name_to_viewer_coords[layer_name] = []
                for source in s.layers[layer_name].source:
                    if source.transform is None:
                        output_dims_final = copy_coord_space(output_dims, "2")
                    else:
                        # Strip channel dims (c^, c', c#) from the existing transform's
                        # output dimensions before priming — channel dims are local to
                        # the layer and handled by channelDimensions, not the source
                        # transform.
                        existing_out = source.transform.output_dimensions
                        spatial_names = [
                            n
                            for n in existing_out.names
                            if not n.endswith(("'", "^", "#"))
                        ]
                        spatial_indices = [
                            list(existing_out.names).index(n) for n in spatial_names
                        ]
                        spatial_space = neuroglancer.CoordinateSpace(
                            names=spatial_names,
                            units=[existing_out.units[i] for i in spatial_indices],
                            scales=np.array(
                                [existing_out.scales[i] for i in spatial_indices]
                            ),
                        )
                        output_dims_final = copy_coord_space(spatial_space, "2")
                    new_coord_space = neuroglancer.CoordinateSpaceTransform(
                        output_dimensions=output_dims_final,
                    )
                    self.stored_map_moving_name_to_viewer_coords[layer_name].append(
                        new_coord_space
                    )
                    source.transform = new_coord_space
        return self.pipeline_state

    def continue_workflow(self, _):
        if self.pipeline_state == PipelineState.NOT_READY:
            all_compatible = self._check_all_moving_layers_are_image_or_seg(
                self.get_state()
            )
            if not all_compatible:
                return
            self.setup_viewer_after_user_ready()
            return
        elif self.pipeline_state == PipelineState.ERROR:
            self.setup_viewer_after_user_ready()
        elif self.pipeline_state == PipelineState.COORDS_READY:
            return
        elif self.pipeline_state == PipelineState.READY:
            with self.viewer.txn() as s:
                for layer_name in self.get_moving_layer_names(s):
                    registered_name = layer_name + "_registered"
                    is_registered_visible = s.layers[registered_name].visible
                    s.layers[registered_name].visible = not is_registered_visible

    def _check_all_moving_layers_are_image_or_seg(self, s: neuroglancer.ViewerState):
        all_images = True
        for layer_name in self.get_moving_layer_names(s):
            layer = s.layers[layer_name]
            if not (layer.type == "image" or layer.type == "segmentation"):
                all_images = False
                break
        if not all_images:
            self._set_status_message(
                "error",
                "All moving layers must be image layers or seg layers for registration to work. Please correct this and try again.",
            )
            self._show_help_message()
        return all_images

    def _show_help_message(self):
        in_prog_message = "Place registration points by moving the centre position of one panel and then putting an annotation with ctrl+left click in the other panel. Annotations can be adjusted if needed with alt+left click. Press 't' to toggle visibility of the registered layer. Press 'f' to toggle forcing at most a similarity transform estimation. Press 'g' to toggle between a local affine estimation and a global one. Press 'd' to download current state for later resumption. Press 'y' to show or hide this help message."
        setup_message = "Place fixed (reference) layers in the left hand panel, and moving layers (to be registered) in the right hand panel. Then press 't' once you have completed this setup. Press 'y' to show/hide this message."
        error_message = (
            f"There was an error in setup. Please try again. {setup_message}"
        )
        waiting_message = "Please wait while setup is completed. In case it seems to be stuck, try pressing 't' again."

        help_message = ""
        if self.pipeline_state == PipelineState.READY:
            help_message = in_prog_message
        elif self.pipeline_state == PipelineState.NOT_READY:
            help_message = setup_message
        elif self.pipeline_state == PipelineState.ERROR:
            help_message = error_message
        elif self.pipeline_state == PipelineState.COORDS_READY:
            help_message = waiting_message
        self._set_status_message("help", help_message)

    def toggle_help_message(self, _):
        help_shown = "help" in self._status_timers
        if help_shown:
            with self.viewer.config_state.txn() as cs:
                self._clear_status_message("help", cs)
        else:
            self._show_help_message()

    def toggle_force_non_affine(self, _):
        self._force_non_affine = not self._force_non_affine
        message = (
            "Estimating max of similarity transformation"
            if self._force_non_affine
            else "Estimating most appropriate transformation"
        )
        self._set_status_message("transform", message)
        self.update_affine()

    def toggle_global_estimate(self, _):
        if self._annotation_filter_method == PointFilter.NONE:
            self._annotation_filter_method = PointFilter.NEAREST
            self._set_status_message(
                "global",
                f"Using nearest {NUM_NEAREST_POINTS} points in transform estimation",
            )
        elif self._annotation_filter_method == PointFilter.NEAREST:
            self._annotation_filter_method = PointFilter.NONE
            self._set_status_message(
                "global", "Using all points in transform estimation"
            )
        self.update_affine()

    def _setup_viewer_actions(self):
        viewer = self.viewer
        continue_name = "continueLinearRegistrationWorkflow"
        viewer.actions.add(continue_name, self.continue_workflow)

        dump_name = "dumpCurrentState"
        viewer.actions.add(dump_name, self.dump_current_state)

        toggle_help_name = "toggleHelpMessage"
        viewer.actions.add(toggle_help_name, self.toggle_help_message)

        force_name = "forceNonAffine"
        viewer.actions.add(force_name, self.toggle_force_non_affine)

        global_name = "toggleGlobalEstimate"
        viewer.actions.add(global_name, self.toggle_global_estimate)

        with viewer.config_state.txn() as cs:
            cs.input_event_bindings.viewer["keyt"] = continue_name
            cs.input_event_bindings.viewer["keyd"] = dump_name
            cs.input_event_bindings.viewer["keyy"] = toggle_help_name
            cs.input_event_bindings.viewer["keyf"] = force_name
            cs.input_event_bindings.viewer["keyg"] = global_name

    def get_moving_layer_names(self, s: neuroglancer.ViewerState):
        right_panel_layers = [
            n for n in s.layout.children[1].layers if n != self.annotations_name
        ]
        return right_panel_layers

    def _copy_moving_layers_to_left_panel(self):
        with self.viewer.txn() as s:
            self._cached_moving_layer_names = self.get_moving_layer_names(s)
            for layer_name in self._cached_moving_layer_names:
                copy = deepcopy(s.layers[layer_name])
                copy.name = layer_name + "_registered"
                copy.visible = False
                s.layers[copy.name] = copy
                s.layout.children[0].layers.append(copy.name)

    def _restore_coord_maps(self, reg_info):
        self._cached_moving_layer_names = self.get_moving_layer_names(self.get_state())
        self.stored_map_moving_name_to_data_coords = {
            k: neuroglancer.CoordinateSpace(json=v)
            for k, v in reg_info["layer_cache"].items()
        }
        self.stored_map_moving_name_to_viewer_coords = {
            k: [neuroglancer.CoordinateSpaceTransform(json_data=t) for t in v]
            for k, v in reg_info["viewer_layer_cache"].items()
        }

    def _handle_layer_names_changed(self, s: neuroglancer.ViewerState):
        current_names = set(self.get_moving_layer_names(s))
        cached_names = set(self.stored_map_moving_name_to_data_coords.keys())
        if current_names == cached_names:
            return
        if len(current_names) == len(cached_names):
            for old_name in cached_names:
                if old_name not in current_names:
                    new_name = list(current_names - cached_names)[0]
                    self.stored_map_moving_name_to_data_coords[new_name] = (
                        self.stored_map_moving_name_to_data_coords.pop(old_name)
                    )
                    self.stored_map_moving_name_to_viewer_coords[new_name] = (
                        self.stored_map_moving_name_to_viewer_coords.pop(old_name)
                    )
                    break
        else:
            self._set_status_message(
                "error",
                "Layers have been added or removed, this may cause unexpected behaviour.",
            )

    def combine_affine_across_dims(self, s: neuroglancer.ViewerState, affine):
        all_dims = s.dimensions.names
        _, moving_dims = self.get_fixed_and_moving_dims(None, all_dims)
        full_matrix = np.zeros((len(all_dims), len(all_dims) + 1))

        for i, dim in enumerate(all_dims):
            for j, dim2 in enumerate(all_dims):
                if dim in moving_dims and dim2 in moving_dims:
                    moving_i = moving_dims.index(dim)
                    moving_j = moving_dims.index(dim2)
                    full_matrix[i, j] = affine[moving_i, moving_j]
                elif dim == dim2:
                    full_matrix[i, j] = 1
            if dim in moving_dims:
                moving_i = moving_dims.index(dim)
                full_matrix[i, -1] = affine[moving_i, -1]
        return full_matrix

    def has_two_coord_spaces(self, s: neuroglancer.ViewerState):
        fixed_dims, moving_dims = self.get_fixed_and_moving_dims(s)
        return len(fixed_dims) == len(moving_dims)

    @debounce(1.5)
    def update_affine(self):
        with self.viewer.txn() as s:
            self._handle_layer_names_changed(s)
            updated = self.estimate_affine(s)
            if updated:
                num_point_pairs = len(self.stored_points[0])
                self.update_registered_layers(s)
                self._set_status_message(
                    "info",
                    f"Estimated affine transform with {num_point_pairs} point pairs",
                )
                if DEBUG:
                    pprint(self.get_registration_info(s))

    def get_fixed_and_moving_dims(
        self, s: neuroglancer.ViewerState | None, dim_names: list | tuple = ()
    ):
        if s is None:
            dimensions = dim_names
        else:
            dimensions = s.dimensions.names
        moving_dims = []
        fixed_dims = []
        for dim in dimensions:
            if dim in NON_PHYSICAL_DIM_NAMES:
                continue
            if dim[:-1] in dimensions:
                moving_dims.append(dim)
            else:
                fixed_dims.append(dim)
        return fixed_dims, moving_dims

    def split_points_into_pairs(self, annotations, dim_names, current_position=None):
        if len(annotations) == 0:
            return np.zeros((0, 0)), np.zeros((0, 0)), None
        first_name = dim_names[0]
        fixed_dims, _ = self.get_fixed_and_moving_dims(None, dim_names)
        real_dims_last = first_name not in fixed_dims
        num_points = len(annotations)
        num_dims = len(annotations[0].point) // 2
        fixed_points = np.zeros((num_points, num_dims))
        moving_points = np.zeros((num_points, num_dims))
        for i, a in enumerate(annotations):
            for j in range(num_dims):
                fixed_index = j + num_dims if real_dims_last else j
                moving_index = j if real_dims_last else j + num_dims
                fixed_points[i, j] = a.point[fixed_index]
                moving_points[i, j] = a.point[moving_index]
        if current_position is not None:
            dim_add = num_dims if real_dims_last else 0
            fixed_position_indices = [i + dim_add for i in range(num_dims)]
            return (
                np.array(fixed_points),
                np.array(moving_points),
                current_position[fixed_position_indices],
            )
        return np.array(fixed_points), np.array(moving_points), current_position

    def update_registered_layers(self, s: neuroglancer.ViewerState):
        if self.affine is not None:
            transform = self.affine.tolist()
            for k, v in self.stored_map_moving_name_to_data_coords.items():
                for i, source in enumerate(s.layers[k].source):
                    registered_source = s.layers[k + "_registered"].source[i]
                    # Channel dims (c^, c', c#) are local to the layer and handled
                    # by channelDimensions; strip them from the source transform's
                    # output dims so only the spatial primed dims remain.
                    ref_out = source.transform.output_dimensions
                    spatial_out_names = [
                        n for n in ref_out.names if not n.endswith(("'", "^", "#"))
                    ]
                    spatial_out_indices = [
                        list(ref_out.names).index(n) for n in spatial_out_names
                    ]
                    output_dims_primed = neuroglancer.CoordinateSpace(
                        names=spatial_out_names,
                        units=[ref_out.units[i] for i in spatial_out_indices],
                        scales=np.array(
                            [ref_out.scales[i] for i in spatial_out_indices]
                        ),
                    )
                    source.transform = neuroglancer.CoordinateSpaceTransform(
                        input_dimensions=v,
                        output_dimensions=output_dims_primed,
                        matrix=transform,
                    )

                    registered_source.transform = neuroglancer.CoordinateSpaceTransform(
                        input_dimensions=v,
                        output_dimensions=v,
                        matrix=transform,
                    )
            annotation_transform = neuroglancer.CoordinateSpaceTransform(
                input_dimensions=create_coord_space_matching_global_dims(s.dimensions),
                output_dimensions=create_coord_space_matching_global_dims(s.dimensions),
                matrix=self.combine_affine_across_dims(s, self.affine).tolist(),
            )
            s.layers[self.annotations_name].source[0].transform = annotation_transform

            if len(self.stored_points[0]) > 0:
                _, moving_dims = self.get_fixed_and_moving_dims(s)
                if moving_dims:
                    last_fixed = np.array(self.stored_points[0][-1])
                    A = np.array(self.affine)
                    new_primed = A[:, :-1] @ last_fixed + A[:, -1]
                    dim_names = list(s.dimensions.names)
                    pos = list(s.position)
                    for i, dim in enumerate(moving_dims):
                        if i < len(new_primed) and dim in dim_names:
                            pos[dim_names.index(dim)] = float(new_primed[i])
                    s.position = pos

            print(f"Updated affine transform (without channel dimensions): {transform}")

    def estimate_affine(self, s: neuroglancer.ViewerState):
        annotations = s.layers[self.annotations_name].annotations

        if len(annotations) == 0:
            if len(self.stored_points[0]) > 0:
                _, moving_dims = self.get_fixed_and_moving_dims(s)
                n_dims = len(moving_dims)
                affine = np.zeros(shape=(n_dims, n_dims + 1))
                for i in range(n_dims):
                    affine[i][i] = 1
                self.affine = affine
                self.stored_points = ([], [], False)
                return True
            return False

        dim_names = s.dimensions.names
        fixed_points, moving_points, current_position = self.split_points_into_pairs(
            annotations, dim_names, s.position
        )
        fixed_points, moving_points = self._filter_annotations(
            fixed_points, moving_points, current_position
        )

        if (
            len(self.stored_points[0]) == len(fixed_points)
            and len(self.stored_points[1]) == len(moving_points)
            and self.stored_points[-1] == self._force_non_affine
        ):
            if np.all(np.isclose(self.stored_points[0], fixed_points)) and np.all(
                np.isclose(self.stored_points[1], moving_points)
            ):
                return False
        self.affine = estimate_transform(
            fixed_points, moving_points, self._force_non_affine
        )
        self.stored_points = [fixed_points, moving_points, self._force_non_affine]

        return True

    def _filter_annotations(
        self, fixed_points: np.ndarray, moving_points: np.ndarray, position
    ):
        if self._annotation_filter_method == PointFilter.NONE:
            return fixed_points, moving_points
        elif self._annotation_filter_method == PointFilter.NEAREST:
            if len(fixed_points) <= NUM_NEAREST_POINTS:
                return fixed_points, moving_points
            diff = fixed_points - np.asarray(position)
            d2 = np.sum(diff * diff, axis=1)
            nearest_indices = np.argpartition(d2, NUM_NEAREST_POINTS - 1)[
                :NUM_NEAREST_POINTS
            ]
            return fixed_points[nearest_indices], moving_points[nearest_indices]
        return fixed_points, moving_points

    def get_registration_info(self, state: neuroglancer.ViewerState):
        info = {}
        annotations = state.layers[self.annotations_name].annotations
        dim_names = state.dimensions.names
        fixed_points, moving_points, _ = self.split_points_into_pairs(
            annotations, dim_names
        )
        info["annotations"] = annotations
        info["fixedPoints"] = fixed_points.tolist()
        info["movingPoints"] = moving_points.tolist()
        if self.affine is not None:
            transformed_points = transform_points(self.affine, moving_points)
            info["transformedPoints"] = transformed_points.tolist()
            info["affineTransform"] = self.affine.tolist()
        return info

    def dump_current_state(self, _):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"neuroglancer_state_{timestamp}.json"

        state = self.get_state()
        state_dict = state.to_json()

        try:
            info = self.get_registration_info(state)
            info.pop("annotations", None)
            info["layer_cache"] = {
                k: v.to_json()
                for k, v in self.stored_map_moving_name_to_data_coords.items()
            }
            info["viewer_layer_cache"] = {
                k: [t.to_json() for t in v]
                for k, v in self.stored_map_moving_name_to_viewer_coords.items()
            }
            info["timestamp"] = timestamp
            state_dict["linear_reg_pipeline_info"] = info
        except Exception:
            print("Error saving registration log")

        content = json.dumps(state_dict, indent=4)
        _browser_download(filename, content, "application/json")

        self._set_status_message(
            "dump", f"State downloaded as {filename} and can be used to continue later."
        )
        return filename

    def get_state(self):
        with self.viewer.txn() as s:
            return s

    def __str__(self):
        return str(self.get_state())

    def _clear_status_messages(self):
        to_pop = []
        for k, v in self._status_timers.items():
            if k == "help":
                continue
            if time() - v > MESSAGE_DURATION:
                to_pop.append(k)
        if not to_pop:
            return
        with self.viewer.config_state.txn() as cs:
            for k in to_pop:
                self._clear_status_message(k, cs)

    def _clear_status_message(self, key: str, config):
        config.status_messages.pop(key, None)
        return self._status_timers.pop(key, None)

    def _set_status_message(self, key: str, message: str):
        with self.viewer.config_state.txn() as cs:
            cs.status_messages[key] = message
        self._status_timers[key] = time()

    def _add_demo_data_to_viewer(self):
        fixed_layer = _create_demo_fixed_image()
        moving_layer = _create_demo_moving_image()

        with self.viewer.txn() as s:
            s.layers["fixed"] = fixed_layer
            s.layers["moving"] = moving_layer

    def _ignore_non_display_dims(self, state: neuroglancer.ViewerState):
        dim_names = state.dimensions.names
        dim_map = {k: 0 for k in dim_names if k not in ["t", "time", "t1"]}
        state.layers[self.annotations_name].clip_dimensions_weight = dim_map


# ---------------------------------------------------------------------------
# Entry point: executed directly by pyodide_worker.ts
# ---------------------------------------------------------------------------

# Check whether the main thread provided a starting Neuroglancer URL.
_starting_url = getattr(js.globalThis, "neuroglancer_starting_url", None)
_starting_state = None
if _starting_url:
    from neuroglancer.url_state import parse_url as _parse_url

    try:
        _starting_state = _parse_url(_starting_url)
        print("Loaded starting state from URL.")
    except Exception as _e:
        print(f"Warning: could not parse starting URL: {_e}")

demo = LinearRegistrationWorkflow(starting_ng_state=_starting_state)
print("Linear registration workflow ready. Press 't' to begin setup.")
