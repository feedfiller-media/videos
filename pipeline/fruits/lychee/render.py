import numpy as np, moderngl
from PIL import Image
from phys import BOARD_HALF
from geom import RM

W, H = 1080, 1920

def perspective(fovy, aspect, n, f):
    t = 1 / np.tan(np.radians(fovy) / 2)
    return np.array([[t / aspect, 0, 0, 0], [0, t, 0, 0], [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)], [0, 0, -1, 0]], np.float32)

def ortho(l, r, b, t, n, f):
    return np.array([[2 / (r - l), 0, 0, -(r + l) / (r - l)], [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                     [0, 0, -2 / (f - n), -(f + n) / (f - n)], [0, 0, 0, 1]], np.float32)

def lookat(eye, tgt, up=(0, 0, 1)):
    eye, tgt, up = map(lambda a: np.array(a, np.float32), (eye, tgt, up))
    f = tgt - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M

COMMON = """
uniform mat4 lightVP; uniform sampler2DShadow shadow; uniform vec3 camPos; uniform vec3 Ldir;
float hash(vec3 p){ p=fract(p*0.3183099+.1); p*=17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
float noise(vec3 x){ vec3 i=floor(x); vec3 f=fract(x); f=f*f*(3.0-2.0*f);
 return mix(mix(mix(hash(i+vec3(0,0,0)),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),
            mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z); }
float fbm(vec3 p){ float a=0.5,s=0.0; for(int i=0;i<4;i++){ s+=a*noise(p); p*=2.03; a*=0.5;} return s; }
float shade_shadow(vec3 wp, vec3 n){
  vec4 lp = lightVP*vec4(wp + n*0.002,1.0); vec3 c = lp.xyz/lp.w*0.5+0.5;
  if(c.x<0.||c.x>1.||c.y<0.||c.y>1.) return 1.0;
  float s=0.0; vec2 ts=1.0/vec2(textureSize(shadow,0));
  for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++) s+=texture(shadow, vec3(c.xy+vec2(x,y)*ts*1.5, c.z-0.0015));
  return s/9.0; }
float spot(vec3 wp){ return smoothstep(1.25, 0.15, length(wp.xy - vec2(0.0, -0.05))); }
vec3 light(vec3 base, vec3 wp, vec3 n, float spec, float gloss){
  vec3 V = normalize(camPos-wp); float sh = shade_shadow(wp,n);
  float diff = max(dot(n,Ldir),0.0);
  vec3 Hh = normalize(Ldir+V); float sp = pow(max(dot(n,Hh),0.0), gloss)*spec;
  float fill = 0.5+0.5*n.z;
  float rim = pow(1.0-max(dot(n,V),0.0),3.0)*0.12;
  float sp2 = spot(wp);
  vec3 c = base*(0.30*fill + 0.95*diff*sh) + vec3(1.0,0.97,0.92)*sp*sh + rim;
  return c*(0.25+0.85*sp2);
}
"""

VS_SCENE = """#version 330
uniform mat4 VP; in vec3 pos; in vec3 nrm; in vec3 rest; in float cut;
out vec3 wp; out vec3 wn; out vec3 rp; out float fc;
void main(){ wp=pos; wn=nrm; rp=rest; fc=cut; gl_Position=VP*vec4(pos,1.0); }"""

FS_MELON = """#version 330
""" + COMMON + """
uniform vec4 planes[100]; uniform int nShown; uniform float RM; uniform float lineA; uniform vec3 gStep; uniform vec3 gLo; uniform vec3 gFront; uniform vec3 gOn;
in vec3 wp; in vec3 wn; in vec3 rp; in float fc; out vec4 o;
void main(){
  vec3 n = normalize(wn); vec3 p = rp/RM; float rr = length(p);
  vec3 col; float spec, gloss; float hb = 0.0; float hk = 0.0;
  if(fc>3.5){
    vec3 lc = fc > 4.5 ? vec3(0.42,0.36,0.14) : mix(vec3(0.10,0.34,0.06), vec3(0.24,0.52,0.12), fbm(wp*90.0));
    float vein = 1.0 - smoothstep(0.0, 0.25, abs(rp.y));
    lc = mix(lc, lc*1.35, vein*step(fc, 4.5)*0.6);
    vec3 nn = n; if(dot(nn, normalize(camPos-wp)) < 0.0) nn = -nn;
    o = vec4(light(lc, wp, nn, 0.5, 40.0) + lc*0.06, 1.0); return; }
  if(fc>2.5){ o = vec4(light(vec3(0.05,0.045,0.045), wp, n, 0.35, 40.0), 1.0); return; }
  if(fc>1.5){
    vec3 V = normalize(camPos-wp); vec3 Rr = reflect(-V, n);
    float env = smoothstep(-0.3, 0.8, Rr.z);
    vec3 steel = mix(vec3(0.48,0.50,0.53), vec3(0.92,0.94,0.97), env) * (0.92 + 0.08*noise(wp*vec3(900.0,40.0,40.0)));
    o = vec4(light(steel, wp, n, 1.6, 140.0) + steel*0.25, 1.0); return; }
  vec3 ps = p; float rs = length(ps);
  if(fc>0.5){
    float rxy = length(ps.yz) / max(sqrt(max(1.0 - ps.x*ps.x, 0.0)), 0.08);
    float lth = atan(ps.z, ps.y);
    float g = fbm(ps*24.0);
    // lychee: glassy white flesh, brown glossy seed, thin pink shell
    float chamber = 0.0;
    float sd = length(vec3(ps.x/0.62, ps.yz/0.36));
    col = mix(vec3(0.94,0.93,0.88), vec3(0.99,0.98,0.95), fbm(ps*25.0));
    col = mix(col, vec3(0.86,0.85,0.80), smoothstep(0.55, 0.95, sd)*(1.0 - smoothstep(1.0, 1.4, sd))*0.4);
    float seed = 1.0 - smoothstep(0.96, 1.0, sd);
    col = mix(col, vec3(0.30,0.15,0.06)*(0.8 + 0.4*fbm(ps*40.0)), seed);
    col = mix(col, vec3(0.88,0.62,0.62), smoothstep(0.935, 0.95, rs));
    col = mix(col, vec3(0.62,0.12,0.14), smoothstep(0.965, 0.98, rs));
    hb = 0.3*(fbm(ps*40.0)-0.5); hk = 0.0006;
    spec = 1.3; gloss = 140.0;
  } else {
    vec3 u = normalize(ps);
    float th = atan(u.y,u.x);
    vec3 q = u*7.0; vec3 ci = floor(q); float F1 = 9.0, F2 = 9.0; vec3 bc = vec3(0);
    for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++) for(int z=-1;z<=1;z++){
      vec3 cc = ci + vec3(x,y,z); vec3 pt = normalize(cc + 0.5 + 0.4*(vec3(hash(cc),hash(cc+1.3),hash(cc+2.7))-0.5))*7.0;
      float dd = length(q-pt); if(dd < F1){ F2 = F1; F1 = dd; bc = cc; } else if(dd < F2) F2 = dd; }
    float crack = 1.0 - smoothstep(0.0, 0.06, F2 - F1);              // grooves between the plates
    float peak = 1.0 - smoothstep(0.0, 0.42, F1);                     // small pointed bump in each plate
    float tone = hash(bc + 4.0);
    col = mix(vec3(0.80,0.10,0.18), vec3(0.98,0.36,0.42), fbm(u*3.0)*0.7 + tone*0.3);
    col = mix(col, vec3(0.86,0.55,0.32), smoothstep(0.55, 0.85, fbm(u*2.5 + 3.0))*0.45);   // ripening yellow patches
    col = mix(col, vec3(0.50,0.08,0.10), crack*0.5);
    col = mix(col, vec3(0.45,0.20,0.10), peak*peak*0.55);                                 // darker tips of the bumps
    hb = 1.1*peak*peak - 0.9*crack + 0.12*(fbm(u*70.0)-0.5); hk = 0.0035;
    float ang = acos(clamp(u.x,-1.0,1.0)); float cal = 1.0 - smoothstep(0.10, 0.16, ang);
    col = mix(col, vec3(0.42,0.30,0.12), cal*0.8);
    spec = 0.35; gloss = 30.0;
    // progressive cut lines (slices so far)
    for(int a=0;a<3;a++){
      if(gOn[a] < 0.5) continue;
      if(a < 2 && rp[a] > gFront[a]) continue;
      if(a == 2 && rp[a] < gFront[a]) continue;     // horizontal cuts go top-down
      float d = (rp[a] - gLo[a]) / gStep[a]; float f = abs(fract(d + 0.5) - 0.5) * gStep[a];
      float l = 1.0 - smoothstep(0.0, min(0.0011, 0.2*gStep[a]), f);
      col = mix(col, vec3(1.0,0.93,0.92), l * 0.55);
    }
    for(int k=0;k<100;k++){ if(k>=nShown) break;
      float d = abs(dot(rp, planes[k].xyz) - planes[k].w);
      col = mix(col, vec3(1.0,0.93,0.92), (1.0 - smoothstep(0.0007,0.0016,d))*lineA);
    }
  }
  if(hk > 0.0){   // screen-space bump mapping (Mikkelsen): real relief from the height field
    vec3 dpx = dFdx(wp), dpy = dFdy(wp); float dhx = dFdx(hb*hk), dhy = dFdy(hb*hk);
    vec3 r1 = cross(dpy, n), r2 = cross(n, dpx); float det = dot(dpx, r1);
    vec3 grad = sign(det)*(dhx*r1 + dhy*r2); n = normalize(abs(det)*n - grad);
    float ao = 0.72 + 0.28*clamp(hb + 1.0, 0.0, 1.0); col *= ao; }
  o = vec4(light(col, wp, n, spec, gloss), 1.0);
}"""

FS_BOARD = """#version 330
""" + COMMON + """
uniform sampler2D texTop; uniform sampler2D texFront; uniform vec3 BH;
in vec3 wp; in vec3 wn; in vec3 rp; in float fc; out vec4 o;
void main(){
  vec3 n = normalize(wn);
  float g = fbm(vec3(wp.x*6.0, wp.y*90.0, wp.z*90.0));
  float grain = sin(wp.y*420.0 + g*9.0 + fbm(wp*25.0)*4.0)*0.5+0.5;
  vec3 base = mix(vec3(0.62,0.40,0.21), vec3(0.47,0.27,0.13), grain*0.6 + g*0.3);
  base *= 0.9 + 0.2*fbm(wp*140.0);
  float burn = 0.0;
  if(n.z>0.9){ vec2 uv = vec2((wp.x+BH.x)/(2.0*BH.x), (wp.y+BH.y)/(2.0*BH.y)); burn = texture(texTop, uv).r; }
  else if(n.y<-0.9){ vec2 uv = vec2((wp.x+BH.x)/(2.0*BH.x), wp.z/(2.0*BH.z)); burn = texture(texFront, uv).r; }
  base = mix(base, vec3(0.13,0.06,0.03), burn*0.9);
  float edge = 0.0;
  o = vec4(light(base, wp, n, 0.12*(1.0-burn), 30.0), 1.0);
}"""

FS_FLOOR = """#version 330
""" + COMMON + """
in vec3 wp; in vec3 wn; in vec3 rp; in float fc; out vec4 o;
void main(){
  if(fc > 0.5){                                   // tiled kitchen wall
    vec2 uv = vec2(wp.x, wp.z) / vec2(0.15, 0.075);
    vec2 id = floor(uv + vec2(0.5*mod(floor(uv.y), 2.0), 0.0)); vec2 f = fract(uv + vec2(0.5*mod(floor(uv.y), 2.0), 0.0));
    float grout = 1.0 - smoothstep(0.0, 0.05, min(min(f.x, 1.0 - f.x) * 2.0, min(f.y, 1.0 - f.y)));
    vec3 tile = vec3(0.30,0.34,0.31) * (0.85 + 0.3*hash(vec3(id, 1.0)));
    vec3 c = mix(tile, vec3(0.07,0.07,0.07), grout);
    float lite = 0.55 + 0.6*exp(-pow(length(vec2(wp.x, wp.z - 0.45)) / 0.75, 2.0));
    o = vec4(c * lite, 1.0); return; }
  vec3 n = vec3(0,0,1);
  float g = fbm(vec3(wp.xy*40.0,0.0))*0.6 + fbm(vec3(wp.xy*160.0,1.0))*0.4;
  vec3 base = vec3(0.22,0.21,0.20)*(0.75+0.5*g);
  vec3 c = light(base, wp, n, 0.15, 30.0);
  o = vec4(c,1.0);
}"""

VS_DEPTH = """#version 330
uniform mat4 VP; in vec3 pos; void main(){ gl_Position = VP*vec4(pos,1.0); }"""
FS_DEPTH = """#version 330
void main(){}"""

def box_mesh(half, center):
    hx, hy, hz = half
    faces = []
    for ax in range(3):
        for sgn in (-1, 1):
            n = np.zeros(3); n[ax] = sgn
            a1, a2 = [k for k in range(3) if k != ax]
            corners = []
            for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3); p[ax] = sgn * half[ax]; p[a1] = u * half[a1]; p[a2] = v * half[a2]
                corners.append(p + center)
            c = np.array(corners)
            tri = [c[0], c[1], c[2], c[0], c[2], c[3]]
            # fix winding
            if np.dot(np.cross(c[1] - c[0], c[2] - c[0]), n) < 0:
                tri = [c[0], c[2], c[1], c[0], c[3], c[2]]
            for t in tri: faces.append(np.concatenate([t, n, t, [0]]))
    return np.array(faces, np.float32)

class Renderer:
    def __init__(self, cam_eye, cam_tgt, fov):
        self.ctx = moderngl.create_standalone_context(backend='egl')
        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.p_melon = ctx.program(vertex_shader=VS_SCENE, fragment_shader=FS_MELON)
        self.p_board = ctx.program(vertex_shader=VS_SCENE, fragment_shader=FS_BOARD)
        self.p_floor = ctx.program(vertex_shader=VS_SCENE, fragment_shader=FS_FLOOR)
        self.p_depth = ctx.program(vertex_shader=VS_DEPTH, fragment_shader=FS_DEPTH)
        self.fov = fov
        self.VP = perspective(fov, W / H, 0.05, 20) @ lookat(cam_eye, cam_tgt)
        L = np.array([-0.22, -0.30, 1.0], np.float32); L /= np.linalg.norm(L)
        self.L = L
        self.LVP = ortho(-0.75, 0.75, -0.75, 0.75, 0.1, 4.0) @ lookat(L * 2.0, (0, 0, 0), (0, 1, 0))
        self.eye = np.array(cam_eye, np.float32)
        # static geometry
        bh = np.array(BOARD_HALF)
        self.board = box_mesh(bh, np.array([0, 0, bh[2]]))
        fl = 12.0
        fq = np.array([[-fl, -fl, 0], [fl, -fl, 0], [fl, fl, 0], [-fl, -fl, 0], [fl, fl, 0], [-fl, fl, 0]], np.float32)
        floor = np.concatenate([fq, np.tile([0, 0, 1], (6, 1)), fq, np.zeros((6, 1))], 1)
        wy = 0.55; wq = np.array([[-fl, wy, 0], [fl, wy, 3], [fl, wy, 0], [-fl, wy, 0], [-fl, wy, 3], [fl, wy, 3]], np.float32)
        wall = np.concatenate([wq, np.tile([0, -1, 0], (6, 1)), wq, np.ones((6, 1))], 1)
        wall2 = wall[[0, 2, 1, 3, 5, 4]]
        self.floor = np.concatenate([floor, wall, wall2]).astype(np.float32)
        self.vb_board = ctx.buffer(self.board.tobytes()); self.vb_floor = ctx.buffer(self.floor.tobytes())
        fmt = ('3f 3f 3f 1f', 'pos', 'nrm', 'rest', 'cut')
        self.va_board = ctx.vertex_array(self.p_board, [(self.vb_board, *fmt)], skip_errors=True)
        self.va_floor = ctx.vertex_array(self.p_floor, [(self.vb_floor, *fmt)], skip_errors=True)
        self.vd_board = ctx.vertex_array(self.p_depth, [(self.vb_board, '3f 28x', 'pos')])
        self.vb_melon = None
        # textures
        def tex(path):
            im = np.array(Image.open(path).convert("L"))[::-1].copy()
            t = ctx.texture((im.shape[1], im.shape[0]), 1, im.tobytes()); t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR); t.anisotropy = 8
            return t
        self.t_top = tex("tex_top.png"); self.t_front = tex("tex_front.png")
        # shadow map
        self.sm = ctx.depth_texture((2048, 2048)); self.sm.compare_func = '<='
        self.sm.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fb_sm = ctx.framebuffer(depth_attachment=self.sm)
        # main fb (MSAA) + resolve
        self.fb_ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4, samples=4)],
                                     depth_attachment=ctx.depth_renderbuffer((W, H), samples=4))
        self.fb_out = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4)])
        for p in (self.p_melon, self.p_board, self.p_floor):
            p['VP'].write(self.VP.T.tobytes()); p['lightVP'].write(self.LVP.T.tobytes())
            p['camPos'].value = tuple(self.eye); p['Ldir'].value = tuple(L); p['shadow'].value = 0
        self.p_board['texTop'].value = 1; self.p_board['texFront'].value = 2
        self.p_board['BH'].value = tuple(bh)
        if 'RM' in self.p_melon: self.p_melon['RM'].value = RM

    def set_grid(self, step, lo, front, on):
        P = self.p_melon
        P['gStep'].value = tuple(map(float, step)); P['gLo'].value = tuple(map(float, lo))
        P['gFront'].value = tuple(map(float, front)); P['gOn'].value = tuple(map(float, on))

    def set_planes(self, planes):
        arr = np.zeros((100, 4), np.float32)
        if planes: arr[:len(planes)] = planes
        self.p_melon['planes'].write(arr.tobytes())

    def render(self, melon_verts, n_shown):
        ctx = self.ctx
        data = melon_verts.astype(np.float32).tobytes()
        if self.vb_melon is None or self.vb_melon.size < len(data):
            if self.vb_melon is not None: self.vb_melon.release()
            self.vb_melon = ctx.buffer(reserve=max(len(data), 1 << 20), dynamic=True)
            fmt = ('3f 3f 3f 1f', 'pos', 'nrm', 'rest', 'cut')
            self.va_melon = ctx.vertex_array(self.p_melon, [(self.vb_melon, *fmt)], skip_errors=True)
            self.vd_melon = ctx.vertex_array(self.p_depth, [(self.vb_melon, '3f 28x', 'pos')])
        self.vb_melon.write(data)
        nv = len(melon_verts)
        # shadow pass
        self.fb_sm.use(); self.fb_sm.clear(depth=1.0)
        self.p_depth['VP'].write(self.LVP.T.tobytes())
        ctx.disable(moderngl.CULL_FACE)
        self.vd_board.render()
        self.vd_melon.render(vertices=nv)
        ctx.enable(moderngl.CULL_FACE)
        # main
        self.fb_ms.use(); self.fb_ms.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.sm.use(0); self.t_top.use(1); self.t_front.use(2)
        self.p_melon['nShown'].value = int(n_shown)
        self.va_floor.render(vertices=18); self.va_board.render(); self.va_melon.render(vertices=nv)
        ctx.copy_framebuffer(self.fb_out, self.fb_ms)
        img = np.frombuffer(self.fb_out.read(components=3), np.uint8).reshape(H, W, 3)[::-1]
        return img
