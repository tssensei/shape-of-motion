from __future__ import annotations

"""Offscreen OpenGL textured-mesh renderer for 2D modal video synthesis."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from modal_peak_pick.core.mesh import RegularGridMesh


VERTEX_SHADER_SOURCE = """
#version 330 core
layout (location = 0) in vec3 in_pos_ndc;
layout (location = 1) in vec2 in_uv;

out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos_ndc, 1.0);
    v_uv = in_uv;
}
"""


FRAGMENT_SHADER_SOURCE = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_texture;

void main() {
    frag_color = texture(u_texture, v_uv);
}
"""


@dataclass(frozen=True)
class GLRenderConfig:
    width: int
    height: int
    show_mesh: bool = False


def image_vertices_to_ndc(
    vertices_xy: np.ndarray,
    image_shape: tuple[int, int],
    vertex_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert image-space mesh vertices into OpenGL NDC coordinates."""
    if vertices_xy.ndim != 2 or vertices_xy.shape[1] != 2:
        raise ValueError("vertices_xy must have shape [N, 2].")

    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 1 or w <= 1:
        raise ValueError("image_shape must be at least 2x2.")

    x = vertices_xy[:, 0].astype(np.float32)
    y = vertices_xy[:, 1].astype(np.float32)
    x_ndc = (x / float(w)) * 2.0 - 1.0
    y_ndc = 1.0 - (y / float(h)) * 2.0
    if vertex_depth is None:
        z_ndc = np.zeros_like(x_ndc, dtype=np.float32)
    else:
        if vertex_depth.ndim != 1 or vertex_depth.shape[0] != vertices_xy.shape[0]:
            raise ValueError("vertex_depth must have shape [N].")
        z_ndc = vertex_depth.astype(np.float32)
    return np.stack([x_ndc, y_ndc, z_ndc], axis=1).astype(np.float32)


class GLMeshWarpRenderer:
    """Hidden-window OpenGL renderer for textured regular-mesh warping."""

    def __init__(self, width: int, height: int, *, show_mesh: bool = False) -> None:
        self.width = int(width)
        self.height = int(height)
        if self.width <= 1 or self.height <= 1:
            raise ValueError("Renderer dimensions must be at least 2x2.")
        self.show_mesh = bool(show_mesh)

        self._gl = None
        self._glfw = None
        self._window = None
        self._program = None
        self._vao = None
        self._vbo_pos = None
        self._vbo_uv = None
        self._ebo = None
        self._tex = None
        self._fbo = None
        self._color_tex = None
        self._depth_rbo = None
        self._index_count = 0
        self._initialized = False

        self._init_context()
        self._init_gl_objects()

    def close(self) -> None:
        if not self._initialized:
            return
        gl = self._gl
        if self._fbo is not None:
            gl.glDeleteFramebuffers(1, [self._fbo])
        if self._color_tex is not None:
            gl.glDeleteTextures(1, [self._color_tex])
        if self._depth_rbo is not None:
            gl.glDeleteRenderbuffers(1, [self._depth_rbo])
        if self._tex is not None:
            gl.glDeleteTextures(1, [self._tex])
        if self._ebo is not None:
            gl.glDeleteBuffers(1, [self._ebo])
        if self._vbo_uv is not None:
            gl.glDeleteBuffers(1, [self._vbo_uv])
        if self._vbo_pos is not None:
            gl.glDeleteBuffers(1, [self._vbo_pos])
        if self._vao is not None:
            gl.glDeleteVertexArrays(1, [self._vao])
        if self._program is not None:
            gl.glDeleteProgram(self._program)

        glfw = self._glfw
        if self._window is not None:
            glfw.destroy_window(self._window)
        glfw.terminate()
        self._initialized = False

    def __enter__(self) -> "GLMeshWarpRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def render(
        self,
        reference_frame_bgr: np.ndarray,
        mesh: RegularGridMesh,
        deformed_vertices_xy: np.ndarray,
        *,
        vertex_depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render one warped frame from a reference texture and deformed mesh."""
        if reference_frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Reference frame shape {reference_frame_bgr.shape[:2]} does not match renderer size {(self.height, self.width)}."
            )
        if mesh.image_shape != (self.height, self.width):
            raise ValueError(f"Mesh image shape {mesh.image_shape} does not match renderer size {(self.height, self.width)}.")
        if deformed_vertices_xy.shape != mesh.vertices_xy.shape:
            raise ValueError(
                f"deformed_vertices_xy shape {deformed_vertices_xy.shape} does not match mesh vertices {mesh.vertices_xy.shape}."
            )

        gl = self._gl
        self._glfw.make_context_current(self._window)

        pos_ndc = image_vertices_to_ndc(deformed_vertices_xy, mesh.image_shape, vertex_depth=vertex_depth)
        uv = mesh.texcoord_uv.astype(np.float32, copy=False)
        indices = mesh.triangles.astype(np.uint32, copy=False).ravel()

        self._upload_reference_texture(reference_frame_bgr)
        self._upload_mesh_buffers(pos_ndc, uv, indices)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glViewport(0, 0, self.width, self.height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glClearDepth(1.0)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glUseProgram(self._program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex)
        gl.glBindVertexArray(self._vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self._index_count, gl.GL_UNSIGNED_INT, None)

        pixels = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        frame_rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        frame_rgb = np.flipud(frame_rgb)
        frame_bgr = frame_rgb[..., ::-1].copy()
        if self.show_mesh:
            frame_bgr = self._overlay_mesh(frame_bgr, mesh, deformed_vertices_xy)

        gl.glBindVertexArray(0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return frame_bgr

    def _overlay_mesh(
        self,
        frame_bgr: np.ndarray,
        mesh: RegularGridMesh,
        deformed_vertices_xy: np.ndarray,
    ) -> np.ndarray:
        overlay = frame_bgr.copy()
        rows, cols = mesh.grid_shape
        pts = np.round(deformed_vertices_xy).astype(np.int32).reshape(rows, cols, 2)
        pts[..., 0] = np.clip(pts[..., 0], 0, self.width - 1)
        pts[..., 1] = np.clip(pts[..., 1], 0, self.height - 1)

        line_color = (40, 220, 255)
        for r in range(rows):
            for c in range(cols - 1):
                cv2.line(overlay, tuple(pts[r, c]), tuple(pts[r, c + 1]), line_color, 1, cv2.LINE_AA)
        for c in range(cols):
            for r in range(rows - 1):
                cv2.line(overlay, tuple(pts[r, c]), tuple(pts[r + 1, c]), line_color, 1, cv2.LINE_AA)
        return overlay

    def _init_context(self) -> None:
        import glfw

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(self.width, self.height, "offscreen", None, None)
        if window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create hidden GLFW window.")

        glfw.make_context_current(window)

        from OpenGL import GL

        self._glfw = glfw
        self._gl = GL
        self._window = window

    def _init_gl_objects(self) -> None:
        gl = self._gl

        self._program = self._create_program(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
        self._vao = gl.glGenVertexArrays(1)
        self._vbo_pos = gl.glGenBuffers(1)
        self._vbo_uv = gl.glGenBuffers(1)
        self._ebo = gl.glGenBuffers(1)
        self._tex = gl.glGenTextures(1)
        self._color_tex = gl.glGenTextures(1)
        self._fbo = gl.glGenFramebuffers(1)
        self._depth_rbo = gl.glGenRenderbuffers(1)

        gl.glBindVertexArray(self._vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 0, None, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_uv)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 0, None, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, 0, None, gl.GL_STATIC_DRAW)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._color_tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, self.width, self.height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._color_tex, 0)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._depth_rbo)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.width, self.height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self._depth_rbo)
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"OpenGL framebuffer is incomplete: status={status}.")

        gl.glUseProgram(self._program)
        gl.glUniform1i(gl.glGetUniformLocation(self._program, "u_texture"), 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindVertexArray(0)
        self._initialized = True

    def _upload_reference_texture(self, frame_bgr: np.ndarray) -> None:
        gl = self._gl
        frame_rgb = frame_bgr[..., ::-1].astype(np.uint8, copy=False)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, self.width, self.height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame_rgb)

    def _upload_mesh_buffers(self, pos_ndc: np.ndarray, uv: np.ndarray, indices: np.ndarray) -> None:
        gl = self._gl
        self._index_count = int(indices.size)

        gl.glBindVertexArray(self._vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, pos_ndc.nbytes, pos_ndc, gl.GL_DYNAMIC_DRAW)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_uv)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, uv.nbytes, uv, gl.GL_STATIC_DRAW)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

    def _create_program(self, vert_source: str, frag_source: str) -> int:
        gl = self._gl
        vs = self._compile_shader(vert_source, gl.GL_VERTEX_SHADER)
        fs = self._compile_shader(frag_source, gl.GL_FRAGMENT_SHADER)
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vs)
        gl.glAttachShader(program, fs)
        gl.glLinkProgram(program)

        linked = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
        if not linked:
            log = gl.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to link OpenGL program: {log}")

        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return program

    def _compile_shader(self, source: str, shader_type: int) -> int:
        gl = self._gl
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        compiled = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if not compiled:
            log = gl.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to compile OpenGL shader: {log}")
        return shader
