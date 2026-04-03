/** @type {import("next").NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "/lab",
  assetPrefix: "/lab",
  trailingSlash: true,
  images: {
    unoptimized: true
  }
};

export default nextConfig;
