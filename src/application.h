#pragma once
#include "vk/vk_window.h"
#include <stdexcept>

#define THROW_IF_FALSE(cond) do { if (!(cond)) throw std::runtime_error(#cond " failed"); } while (0)

// Partially adapted from my own open source project QhenkiX
// Self plug: https://github.com/AaronTian-stack/QhenkiX
class Application
{
protected:
	pt::VulkanWindow m_window;
	pt::VulkanContext m_context;
	bool m_quit = false;
	// TODO: semaphores

	virtual void init_window();
	virtual void create() {}
	virtual void render() {}
	virtual void destroy() {}
public:
	void run(bool enable_debug_layer);
	virtual ~Application() = 0;
};
