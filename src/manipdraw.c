/*
 * Copyright 2023 Saso Kiselkov. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * “Software”), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 * NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdbool.h>
#include <time.h>

#include <XPLMDisplay.h>
#include <XPLMPlugin.h>
#include <XPLMPlanes.h>

#include <cglm/cglm.h>

#include <acfutils/acf_file.h>
#include <acfutils/crc64.h>
#include <acfutils/dr.h>
#include <acfutils/glew.h>
#include <acfutils/log.h>
#include <acfutils/helpers.h>
#include <acfutils/osrand.h>
#include <acfutils/shader.h>

#include <obj8.h>

#include <vector>
#include <string>

#define	PLUGIN_NAME		"manipdraw"
#define	PLUGIN_SIG		"skiselkov.manipdraw"
#define	PLUGIN_DESCRIPTION	"manipdraw"

typedef struct _manip_click {
	uint16_t index;
	uint64_t start_t; 
} manip_click ;



/* obj8_t structure...

	
	n_manips (count of manipulators)
	manips (array of obj8_manip_t objects)

		manip type type


		cmdname

		inside union 

			manip_axis_knob
			struct {
				float		min, max;
				float		d_click, d_hold;
				dr_t		dr;
			} manip_axis_knob;


			XPLMCommandRef		cmd;

			struct {
				vect3_t		d;
				XPLMCommandRef	pos_cmd;
				XPLMCommandRef	neg_cmd;
			} cmd_axis;
			struct {
				XPLMCommandRef	pos_cmd;
				XPLMCommandRef	neg_cmd;
			} cmd_knob;
			struct {
				XPLMCommandRef	pos_cmd;
				XPLMCommandRef	neg_cmd;
			} cmd_sw;
			struct {
				float		dx, dy, dz;
				float		v1, v2;
				unsigned	drset_idx;
			} drag_axis;
			struct {
				vect3_t		xyz;
				vect3_t		dir;
				float		angle1, angle2;
				float		lift;
				float		v1min, v1max;
				float		v2min, v2max;
				unsigned	drset_idx1, drset_idx2;
			} drag_rot;
			struct {
				float		dx, dy;
				float		v1min, v1max;
				float		v2min, v2max;
				unsigned	drset_idx1, drset_idx2;
			} drag_xy;
			struct {
				unsigned	drset_idx;
				float		v1, v2;
			} toggle;
*/


static std::vector<manip_click> manip_clicks;


static struct {
	dr_t	fbo;
	dr_t	viewport;
	dr_t	acf_matrix;
	dr_t	mv_matrix;
	dr_t	proj_matrix_3d;
	dr_t	rev_float_z;
	dr_t	modern_drv;
} drs;

static int		xpver = 0;
static char		plugindir[512] = { 0 };

static GLuint		cursor_tex[2] = {};
static GLuint		cursor_fbo = 0;
static GLuint		cursor_pbo = 0;
static bool		cursor_xfer = false;
static uint16_t		manip_idx = UINT16_MAX;

static uint64_t		last_draw_t = 0;
static uint64_t		blink_start_t = 0;
static uint16_t		prev_manip_idx = UINT16_MAX;

static shader_info_t generic_vert_info = { .filename = "generic.vert.spv" };
static shader_info_t resolve_frag_info = { .filename = "resolve.frag.spv" };
static shader_info_t paint_frag_info = { .filename = "paint.frag.spv" };
static const shader_prog_info_t resolve_prog_info = {
    .progname = "manipdraw_resolve",
    .vert = &generic_vert_info,
    .frag = &resolve_frag_info
};
static const shader_prog_info_t paint_prog_info = {
    .progname = "manipdraw_paint",
    .vert = &generic_vert_info,
    .frag = &paint_frag_info
};
static shader_obj_t	resolve_shader = {};
static shader_obj_t	paint_shader = {};
static obj8_t		*obj = NULL;

unsigned int indexToPaint = -1u;

enum {
    U_PVM,
    U_COLOR,
    NUM_UNIFORMS
};
static const char *uniforms[NUM_UNIFORMS] = {
    [U_PVM] = "pvm",
    [U_COLOR] = "color"
};

static void
resolve_manip_complete(void)
{
	const uint16_t *data;

	/* No transfer in progress, so allow caller to start a new update */
	if (!cursor_xfer)
		return;

	ASSERT(cursor_pbo != 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, cursor_pbo);
	data = (const uint16_t *)glMapBuffer(GL_PIXEL_PACK_BUFFER,
	    GL_READ_ONLY);
	if (data != NULL) {
		/* single pixel containing the clickspot index */
		manip_idx = *data;
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

		if (manip_idx != UINT16_MAX && manip_idx != prev_manip_idx) {

			logMsg("New manip idx is %d", manip_idx);

			prev_manip_idx = manip_idx;

			uint64_t now = microclock();
	
			manip_click click;
			click.index = manip_idx;
			click.start_t = now;

			manip_clicks.push_back(click);

		}
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	cursor_xfer = false;
}

static bool
is_rev_float_z(void)
{
	return (xpver >= 12000 || dr_geti(&drs.modern_drv) != 0 ||
	    dr_geti(&drs.rev_float_z) != 0);
}

static void
resolve_manip(int mouse_x, int mouse_y, const mat4 pvm)
{
	int vp[4];

	ASSERT(pvm != NULL);

	resolve_manip_complete();

	VERIFY3S(dr_getvi(&drs.viewport, vp, 0, 4), ==, 4);

	ASSERT(cursor_fbo != 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER, cursor_fbo);
	glViewport(vp[0] - mouse_x, vp[1] - mouse_y, vp[2], vp[3]);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	if (is_rev_float_z()) {
		glDepthFunc(GL_GREATER);
		glClearDepth(0);
	}
	/*
	 * We want to set the FBO's color to 1, which is 0xFFFF in 16-bit.
	 * That way, if nothing covers it, we know that there is no valid
	 * manipulator there.
	 */
	glClearColor(1, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0, 0, 0, 0);

	shader_obj_bind(&resolve_shader);
	glUniformMatrix4fv(shader_obj_get_u(&resolve_shader, U_PVM),
	    1, GL_FALSE, (const GLfloat *)pvm);
	ASSERT(obj != NULL);
	obj8_set_render_mode(obj, OBJ8_RENDER_MODE_MANIP_ONLY);
	obj8_draw_group(obj, NULL, shader_obj_get_prog(&resolve_shader), pvm);

	ASSERT(cursor_pbo != 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, cursor_pbo);
	glReadPixels(0, 0, 1, 1, GL_RED, GL_UNSIGNED_SHORT, NULL);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	cursor_xfer = true;
	/*
	 * Restore original XP viewport & framebuffer binding.
	 */
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);
	if (is_rev_float_z()) {
		glDepthFunc(GL_LESS);
		glClearDepth(1);
	}
	glBindFramebufferEXT(GL_FRAMEBUFFER, dr_geti(&drs.fbo));
	glViewport(vp[0], vp[1], vp[2], vp[3]);
}


unsigned int countup = 0;

unsigned int todraw = 0;

static void
paint_manip(const mat4 pvm)
{
	uint64_t now = microclock(), delta_t = 0;
	int vp[4];
	float alpha;
	mat4 pvm_in;

	memcpy(pvm_in, pvm, sizeof (pvm));

	ASSERT(pvm != NULL);

	VERIFY3S(dr_getvi(&drs.viewport, vp, 0, 4), ==, 4);

	alpha = 0.5;

	countup++;
	unsigned int newtodraw = (countup / 2) % 200;

	if (newtodraw != todraw) {
		logMsg("[DEBUG] Now drawing by counter %d", newtodraw);
		todraw = newtodraw;
	}


	shader_obj_bind(&paint_shader);
	glUniformMatrix4fv(shader_obj_get_u(&paint_shader, U_PVM),
						1, GL_FALSE, (const GLfloat *)pvm);

	glUniform4f(shader_obj_get_u(&paint_shader, U_COLOR), 1, 0, 0, alpha);
	
	ASSERT(obj != NULL);
	glEnable(GL_BLEND);

	//obj8_set_render_mode2(obj, OBJ8_RENDER_MODE_MANIP_ONLY_ONE, todraw);
	
	//obj8_draw_group(obj, NULL, shader_obj_get_prog(&paint_shader), pvm);


	//obj8_set_render_mode2(obj, OBJ8_RENDER_MODE_NONMANIP_ONLY_ONE, todraw);
	
	//obj8_draw_group(obj, NULL, shader_obj_get_prog(&paint_shader), pvm);


	//glUniform4f(shader_obj_get_u(&paint_shader, U_COLOR), 0, 1, 0, alpha);
	
	logMsg("[DEBUG] Painting cmd idx of %d", indexToPaint);

	obj8_set_render_mode2(obj, OBJ8_RENDER_MODE_NONMANIP_ONLY_ONE, indexToPaint);
	
	//obj8_draw_group(obj, NULL, shader_obj_get_prog(&paint_shader), pvm);

	obj8_draw_group_by_cmdidx(obj, indexToPaint, shader_obj_get_prog(&paint_shader), pvm);

	// obj8_set_render_mode2(obj, OBJ8_RENDER_MODE_MANIP_ONLY, 0);


	// alpha = 0.5;

	// shader_obj_bind(&paint_shader);
	// glUniformMatrix4fv(shader_obj_get_u(&paint_shader, U_PVM),
	// 					1, GL_FALSE, (const GLfloat *)pvm);

	// glUniform4f(shader_obj_get_u(&paint_shader, U_COLOR), 1, 0, 1, alpha);
	
	// ASSERT(obj != NULL);
	// glEnable(GL_BLEND);
	
	// obj8_draw_by_counter(obj, shader_obj_get_prog(&paint_shader), todraw, pvm_in);


	glViewport(vp[0], vp[1], vp[2], vp[3]);

	last_draw_t = now;

	/*
	if (manip_idx != prev_manip_idx || now - last_draw_t > SEC2USEC(0.2)) {
		blink_start_t = now;
		prev_manip_idx = manip_idx;
		logMsg("New manip idx is %d", manip_idx);
	}
	last_draw_t = now;
	delta_t = (now - blink_start_t) % 1000000;
	if (delta_t < 500000)
		alpha = delta_t / 500000.0;
	else
		alpha = 1 - (delta_t - 500000) / 500000.0;

	obj8_set_render_mode2(obj, OBJ8_RENDER_MODE_MANIP_ONLY_ONE, manip_idx);
	obj8_draw_group(obj, NULL, shader_obj_get_prog(&paint_shader), pvm);

	obj8_set_render_mode2(obj, OBJ8_RENDER_MODE_MANIP_ONLY_ONE, 86);
	obj8_draw_group(obj, NULL, shader_obj_get_prog(&paint_shader), pvm);
	*/

	
}

static bool
should_draw_manip(uint16_t manip_idx)
{
	const obj8_manip_t *manip;

	if (manip_idx == UINT16_MAX)
		return (false);
	ASSERT(obj != NULL);
	manip = obj8_get_manip(obj, manip_idx);
	return (manip->type != OBJ8_MANIP_NOOP);
}

static int
draw_cb(XPLMDrawingPhase phase, int before, void *refcon)
{
	int mouse_x, mouse_y;
	int vp[4];
	mat4 proj_matrix, acf_matrix, pvm;

	UNUSED(phase);
	UNUSED(before);
	UNUSED(refcon);

	XPLMGetMouseLocationGlobal(&mouse_x, &mouse_y);
	VERIFY3S(dr_getvi(&drs.viewport, vp, 0, 4), ==, 4);

	if (mouse_x < vp[0] || mouse_x > vp[0] + vp[2] ||
	    mouse_y < vp[1] || mouse_y > vp[1] + vp[3]) {
		/* Mouse off-screen, don't draw anything */
		return (1);
	}
	/*
	 * Mouse is somewhere on the screen. Redraw the manipulator stack.
	 */
	shader_obj_reload_check(&resolve_shader);
	shader_obj_reload_check(&paint_shader);

	dr_getvf32(&drs.acf_matrix, (float *)acf_matrix, 0, 16);
	dr_getvf32(&drs.proj_matrix_3d, (float *)proj_matrix, 0, 16);
	glm_mat4_mul(proj_matrix, acf_matrix, pvm);

	UNUSED(resolve_manip);
	resolve_manip(mouse_x, mouse_y, pvm);
	//if (true || should_draw_manip(manip_idx))
	paint_manip(pvm);
	
	glUseProgram(0);

	return (1);
}

static void
create_cursor_objects(void)
{
	/*
	 * Create the textures which will hold the rendered manipulator
	 * pixel right under the user's cursor spot. We need two textures
	 * here, one to hold the manipulator ID (16-bit single-channel
	 * texture, using the GL_RED channel), and another one to hold
	 * the depth buffer (to properly handle depth and occlusion).
	 */
	glGenTextures(ARRAY_NUM_ELEM(cursor_tex), cursor_tex);
	VERIFY(cursor_tex[0] != 0);
	setup_texture(cursor_tex[0], GL_R16, 1, 1,
	    GL_RED, GL_UNSIGNED_SHORT, NULL);
	setup_texture(cursor_tex[1], GL_DEPTH_COMPONENT32F, 1, 1,
	    GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	/*
	 * Set up the framebuffer object. This will be the target to draw
	 * the manipulator IDs. The contents of the framebuffer will be
	 * backed by the textures created above.
	 */
	glGenFramebuffers(1, &cursor_fbo);
	VERIFY(cursor_fbo != 0);
	setup_color_fbo_for_tex(cursor_fbo, cursor_tex[0], cursor_tex[1], 0,
	    false);
	/*
	 * Set up the back-transfer pixel buffer. This is used to retrieve
	 * the manipulator render result back from GPU VRAM.
	 */
	glGenBuffers(1, &cursor_pbo);
	VERIFY(cursor_pbo != 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, cursor_pbo);
	glBufferData(GL_PIXEL_PACK_BUFFER, sizeof (uint16_t), NULL,
	    GL_STREAM_READ);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

static void
destroy_cursor_objects(void)
{
	if (cursor_pbo != 0) {
		glDeleteBuffers(1, &cursor_pbo);
		cursor_pbo = 0;
	}
	if (cursor_fbo != 0) {
		glDeleteFramebuffers(1, &cursor_fbo);
		cursor_fbo = 0;
	}
	if (cursor_tex[0] != 0) {
		glDeleteTextures(ARRAY_NUM_ELEM(cursor_tex), cursor_tex);
		memset(cursor_tex, 0, sizeof (cursor_tex));
	}
}

static void
log_dbg_string(const char *str)
{
	XPLMDebugString(str);
}

PLUGIN_API int
XPluginStart(char *name, char *sig, char *desc)
{
	char *p;
	GLenum err;
	uint64_t seed;
	int xplm_ver;
	XPLMHostApplicationID host_id;
	/*
	 * libacfutils logging facility bootstrap, this must be one of
	 * the first steps during init, to make sure we have the logMsg
	 * and general error logging facilities available early.
	 */
	log_init(log_dbg_string, "manipdraw");

	ASSERT(name != NULL);
	ASSERT(sig != NULL);
	ASSERT(desc != NULL);
	XPLMGetVersions(&xpver, &xplm_ver, &host_id);
	/*
	 * Always use Unix-native paths on the Mac!
	 */
	XPLMEnableFeature("XPLM_USE_NATIVE_PATHS", 1);
	XPLMEnableFeature("XPLM_USE_NATIVE_WIDGET_WINDOWS", 1);
	/*
	 * Construct plugindir to point to our plugin's root directory.
	 */
	XPLMGetPluginInfo(XPLMGetMyID(), NULL, plugindir, NULL, NULL);
	fix_pathsep(plugindir);
	/* cut off the trailing path component (our filename) */
	if ((p = strrchr(plugindir, DIRSEP)) != NULL)
		*p = '\0';
	/* cut off an optional '32' or '64' trailing component */
	if ((p = strrchr(plugindir, DIRSEP)) != NULL) {
		if (strcmp(p + 1, "64") == 0 || strcmp(p + 1, "32") == 0 ||
		    strcmp(p + 1, "win_x64") == 0 ||
		    strcmp(p + 1, "mac_x64") == 0 ||
		    strcmp(p + 1, "lin_x64") == 0)
			*p = '\0';
	}
	/*
	 * Initialize the CRC64 and PRNG machinery inside of libacfutils.
	 */
	crc64_init();
	if (!osrand(&seed, sizeof (seed)))
		seed = microclock() + clock();
	crc64_srand(seed);
	/*
	 * GLEW bootstrap
	 */
	err = glewInit();
	if (err != GLEW_OK) {
		/* Problem: glewInit failed, something is seriously wrong. */
		logMsg("FATAL ERROR: cannot initialize libGLEW: %s",
		    glewGetErrorString(err));
		goto errout;
	}
	if (!GLEW_VERSION_2_1) {
		logMsg("FATAL ERROR: your system doesn't support OpenGL 2.1");
		goto errout;
	}
	strcpy(name, PLUGIN_NAME);
	strcpy(sig, PLUGIN_SIG);
	strcpy(desc, PLUGIN_DESCRIPTION);

	return (1);
errout:
	return (0);
}

PLUGIN_API void
XPluginStop(void)
{
	log_fini();
}

static inline const char *acf_find_prop(const acf_file_t *acf, const std::string property)
{
	return acf_prop_find(acf, property.c_str());
}

PLUGIN_API int
XPluginEnable(void)
{

	char *shader_dir, *obj_path;

	fdr_find(&drs.fbo, "sim/graphics/view/current_gl_fbo");
	fdr_find(&drs.viewport, "sim/graphics/view/viewport");
	fdr_find(&drs.acf_matrix, "sim/graphics/view/acf_matrix");
	fdr_find(&drs.mv_matrix, "sim/graphics/view/modelview_matrix");
	fdr_find(&drs.proj_matrix_3d, "sim/graphics/view/projection_matrix_3d");
	if (!dr_find(&drs.rev_float_z,
	    "sim/graphics/view/is_reverse_float_z") ||
	    !dr_find(&drs.modern_drv,
	    "sim/graphics/view/using_modern_driver")) {
		ASSERT3S(xpver, >=, 12000);
	}
	
	create_cursor_objects();

	shader_dir = mkpathname(plugindir, "shaders", NULL);

	logMsg("[DEBUG] Will init shaders from path: %s", shader_dir);

	if (!shader_obj_init(&resolve_shader, shader_dir, &resolve_prog_info,
	    NULL, 0, uniforms, NUM_UNIFORMS) ||
	    !shader_obj_init(&paint_shader, shader_dir, &paint_prog_info,
	    NULL, 0, uniforms, NUM_UNIFORMS)) {
		
		lacf_free(shader_dir);
		return (0);
	}

	return (1);
}


void new_aircraft_loaded()
{
	
	VERIFY(XPLMRegisterDrawCallback(draw_cb, xplm_Phase_Window, 1, NULL));


	vect3_t pos_offset = ZERO_VECT3; // moves to right, moves ?, moves back

	std::string _aircraftFolderPath;

    static char aircraftPath[2048];
    static char aircraftFileName[1024];

    XPLMGetNthAircraftModel(0, aircraftFileName, aircraftPath); 

    std::string _aircraftFilePath = aircraftPath;

   	_aircraftFolderPath = aircraftPath;
    std::string key(aircraftFileName);

    std::size_t found = _aircraftFolderPath.rfind(key);
    if (found!=std::string::npos) {
        _aircraftFolderPath.replace(found,key.length(),"");
    } else {
    	assert(false);
    }

	acf_file_t *acf = acf_file_read(_aircraftFilePath.c_str());
    
    bool desired_object_found = false;

	std::string objectFileName = "knobs.obj";

	std::string _aircraftObjectPath;


    if(acf) {

	    const char *obj_in_acf = NULL;

	    unsigned int idx = 0;


	    while(obj_in_acf = acf_find_prop(acf, "_obja/" + std::to_string(idx) + "/_v10_att_file_stl")) {

	    	std::string obj_in_acf_str = std::string(obj_in_acf);

	    	std::size_t found = obj_in_acf_str.find(objectFileName);

	        if (found != std::string::npos) {
	        	desired_object_found = true;

	        	_aircraftObjectPath = _aircraftFolderPath + "/objects/" + obj_in_acf;
	            
	            logMsg("[DEBUG] Found cockpit object at: %s", _aircraftObjectPath.c_str());

	            const char *x_acf_prt = acf_find_prop(acf, "_obja/" + std::to_string(idx) + "/_v10_att_x_acf_prt_ref");
		        const char *y_acf_prt = acf_find_prop(acf, "_obja/" + std::to_string(idx) + "/_v10_att_y_acf_prt_ref");
		        const char *z_acf_prt = acf_find_prop(acf, "_obja/" + std::to_string(idx) + "/_v10_att_z_acf_prt_ref");

		        if (x_acf_prt) {
		        	pos_offset.x = std::stof(x_acf_prt) / 3.2808398950131;
		        }
		        if (y_acf_prt) {
		        	pos_offset.y = std::stof(y_acf_prt) / 3.2808398950131;
		        }
		        if (z_acf_prt) {
		        	pos_offset.z = std::stof(z_acf_prt) / 3.2808398950131;
		        }
		        
	        	break;
	        }

	        idx++;
	    }

   		acf_file_free(acf);

   	}

   	if (!desired_object_found) {
   		assert(false);
   		return;
   	}
   	
	obj = obj8_parse(_aircraftObjectPath.c_str(), pos_offset);

	if (!obj) {
		assert(false);
		return;
	}

	while (obj8_is_load_complete(obj) == false) {

	}


	logMsg("Report on obj cmds found...");

	unsigned n_cmd_t = obj8_get_num_cmd_t(obj);

	for (unsigned i = 0; i < n_cmd_t; i++) {

		const obj8_cmd_t *cmd = obj8_get_cmd_t(obj, i);

		const obj8_drset_t *dr_set_for_obj = obj8_get_drset(obj);

		unsigned drset_idx = obj8_get_cmd_drset_idx(cmd);

		unsigned cmd_idx = obj8_get_cmd_idx(cmd);

		const char *dr_name_for_cmd = obj8_drset_get_dr_name(dr_set_for_obj, drset_idx);

		obj8_debug_cmd(obj, cmd);

		logMsg("Found cmdidx %d has drset idx of %d for %s", cmd_idx, drset_idx, dr_name_for_cmd);

		if (strcmp(dr_name_for_cmd, "ckpt/pushbutton/39") == 0) {
			logMsg("[DEBUG] FOUND INDEX TO PAINT OF %d", i);
			indexToPaint = i;

			logMsg("[DEBUG] Will look for nearest tris...");

			unsigned tris_cmd_idx = obj8_nearest_tris_for_cmd(obj, cmd);

			logMsg("[DEBUG] Found tris cmdidx %d", tris_cmd_idx);

			indexToPaint = tris_cmd_idx;
		}

	}

	logMsg("Found %d manipulators for object", obj8_get_num_manips(obj));

	/* Lets print info from the parsed object... */

	for (unsigned i = 0 ; i < obj8_get_num_manips(obj); i++) {

		/*

		obj->drset[i]

		obj8_drset_get_dr_name(obj->drset, cmd->drset_idx)

		typedef struct {
			unsigned	n_drs;
			avl_tree_t	tree;
			list_t		list;
			bool		complete;
			float		*values;
		} obj8_drset_t;
			*/

		const obj8_manip_t* obj_manip = obj8_get_manip(obj, i);

		switch(obj_manip->type) {

			case OBJ8_MANIP_AXIS_KNOB:
				logMsg("For manip at index %d of type OBJ8_MANIP_AXIS_KNOB the relevent dr is %s",i, obj_manip->manip_axis_knob.dr.name);
				break;
			case OBJ8_MANIP_COMMAND:
				logMsg("For manip at index %d of type OBJ8_MANIP_COMMAND the relevent cmd ref is stored in obj_manip->cmd", i);
				break;
			case OBJ8_MANIP_COMMAND_AXIS:
				logMsg("For manip at index %d of type OBJ8_MANIP_COMMAND_AXIS the relevent cmd refs are pos_cmd and neg_com is stored in obj_manip->cmd_axis.pos_cmd and obj_manip->cmd_axis.neg_cmd", i);
				break;
			case OBJ8_MANIP_COMMAND_KNOB:
				logMsg("For manip at index %d of type OBJ8_MANIP_COMMAND_KNOB the relevent cmd refs are pos_cmd and neg_com is stored in obj_manip->cmd_knob.pos_cmd and obj_manip->cmd_knob.neg_cmd", i);
				break;
			case OBJ8_MANIP_COMMAND_SWITCH_LR:
				logMsg("For manip at index %d of type OBJ8_MANIP_COMMAND_SWITCH_LR the relevent cmd refs are pos_cmd and neg_com is stored in obj_manip->cmd_sw.pos_cmd and obj_manip->cmd_sw.neg_cmd", i);
				break;
			case OBJ8_MANIP_COMMAND_SWITCH_UD:
				logMsg("For manip at index %d of type OBJ8_MANIP_COMMAND_SWITCH_UD the relevent cmd refs are pos_cmd and neg_com is stored in obj_manip->cmd_sw.pos_cmd and obj_manip->cmd_sw.neg_cmd", i);
				break;
			case OBJ8_MANIP_COMMAND_SWITCH_LR2:
				logMsg("For manip at index %d of type OBJ8_MANIP_COMMAND_SWITCH_LR2 the relevent cmd ref is stored in obj_manip->cmd_sw2", i);
				break;
			case OBJ8_MANIP_COMMAND_SWITCH_UD2:
				logMsg("For manip at index %d of type OBJ8_MANIP_COMMAND_SWITCH_UD2 the relevent cmd ref is stored in obj_manip->cmd_sw2", i);
				break;
			case OBJ8_MANIP_DRAG_AXIS:
				logMsg("For manip at index %d of type OBJ8_MANIP_DRAG_AXIS the relevent dr is %s", i, obj8_drset_get_dr_name(obj8_get_drset(obj), obj_manip->drag_axis.drset_idx));
				break;
			case OBJ8_MANIP_DRAG_ROTATE:
				logMsg("For manip at index %d of type OBJ8_MANIP_DRAG_ROTATE the relevent dr's are %s AND %s", i, obj8_drset_get_dr_name(obj8_get_drset(obj), obj_manip->drag_rot.drset_idx1), obj8_drset_get_dr_name(obj8_get_drset(obj), obj_manip->drag_rot.drset_idx2));
				break;
			case OBJ8_MANIP_DRAG_XY:
				logMsg("For manip at index %d of type OBJ8_MANIP_DRAG_XY the relevent dr's are %s AND %s", i, obj8_drset_get_dr_name(obj8_get_drset(obj), obj_manip->drag_xy.drset_idx1), obj8_drset_get_dr_name(obj8_get_drset(obj), obj_manip->drag_xy.drset_idx2));
				break;
			case OBJ8_MANIP_TOGGLE:
				logMsg("For manip at index %d of type OBJ8_MANIP_TOGGLE the relevent dr is %s", i, obj8_drset_get_dr_name(obj8_get_drset(obj), obj_manip->toggle.drset_idx));
				break;
			case OBJ8_MANIP_NOOP:
				logMsg("For manip type of OBJ8_MANIP_NOOP no relevent dr or cmd");
				break;
			default:
				break;
		}
	}
}

PLUGIN_API void
XPluginDisable(void)
{
	XPLMUnregisterDrawCallback(draw_cb, xplm_Phase_Window, 1, NULL);

	destroy_cursor_objects();
	shader_obj_fini(&resolve_shader);
	shader_obj_fini(&paint_shader);
	if (obj != NULL) {
		obj8_free(obj);
		obj = NULL;
	}
}


PLUGIN_API void XPluginReceiveMessage(
XPLMPluginID	inFromWho,
int				inMessage,
void *			inParam)
{
    
    //Here we will
    
    //PRINTF("C Plugin received message.\n");
    
    /*
    int sfd_msg_rcv_val = SF_MSG_RCV;
    
    if (inMessage == sfd_msg_rcv_val) {
        
        SASL_MSG_StringData messageData = *((SASL_MSG_StringData *)inParam);
        
		SFFlightClientController::getInstance().handle_message_from_ui(messageData.mData);
    }*/

    switch (inMessage) {

    	case XPLM_MSG_PLANE_CRASHED:
    		/* This message is sent to your plugin whenever the user's plane crashes.      */

    		break;

    	case XPLM_MSG_PLANE_LOADED:
    		/* This message is sent to your plugin whenever a new plane is loaded.  The    *
			 * parameter is the number of the plane being loaded; 0 indicates the user's   *
			 * plane.                                                                      */
    		
    		//NOTE: This is an absurd aspect of the XPLM that a void* is actually an int!
    		if (inParam == 0) {
    			//logMsg("Aircraft loaded...");
    			new_aircraft_loaded();
    		}

    	default:
    		break;
    }
}