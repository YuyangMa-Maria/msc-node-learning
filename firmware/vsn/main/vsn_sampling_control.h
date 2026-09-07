// Sampling-control interface used by the VSN runtime and host protocol.

#pragma once

bool vsn_sampling_control_init();
void vsn_sampling_set_automatic(bool automatic);
bool vsn_sampling_trigger_once();
bool vsn_sampling_schedule_after_asn();
void vsn_sampling_cancel_after_asn();
void vsn_sampling_notify_asn_ready();
bool vsn_sampling_wait_for_frame();
bool vsn_sampling_is_automatic();
