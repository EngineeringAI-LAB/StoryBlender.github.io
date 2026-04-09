conda list -n mcp | grep -v "^python " | grep -v "^storyblender " > requirements_blender.txt
echo "nest_asyncio              1.6.0                    pypi_0    pypi" >> requirements_blender.txt