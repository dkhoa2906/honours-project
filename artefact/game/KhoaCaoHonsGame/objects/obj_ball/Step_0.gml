/**
if place_meeting(x, y+vspeed, obj_wall) || place_meeting(x, y+vspeed, obj_paddle){
	hspeed = -hspeed
}

if place_meeting(x+hspeed, y, obj_wall) || place_meeting(x+hspeed, y, obj_paddle){
	vspeed = -vspeed
}


if (place_meeting(x + hspeed, y, obj_paddle) || place_meeting(x + hspeed, y, obj_wall)) {
    hspeed = -hspeed;
} else {
    x += hspeed;
}

if (place_meeting(x, y + vspeed, obj_paddle) || place_meeting(x, y + vspeed, obj_wall)) {
    vspeed = -vspeed;
} else {
    y += vspeed;
}
**/


var pad = instance_place(x, y + vspeed, obj_paddle);

if (pad != noone)
{
    while (!place_meeting(x, y + sign(vspeed), obj_paddle))
    {
        y += sign(vspeed);
    }

    var offset = (x - pad.x) / (pad.sprite_width / 2);
    offset = clamp(offset, -1, 1);

    var max_angle = 60;
    var angle = offset * max_angle;

    speed = point_distance(0,0,hspeed,vspeed);

    hspeed = lengthdir_x(speed, angle);
    vspeed = -lengthdir_y(speed, angle);
}