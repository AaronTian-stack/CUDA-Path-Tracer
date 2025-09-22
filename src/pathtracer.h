#pragma once
#include "application.h"

class PathTracer : public Application
{
protected:
	void create() override;
	void render() override;
	void destroy() override;

public:
	~PathTracer() override;
};
