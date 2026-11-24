import { predict } from "./inference.js";

self.onmessage = async ({ data: file }) => {
  try {
    const { matches, seconds } = await predict(file, status => self.postMessage({ status }));
    self.postMessage({ matches, seconds });
  } catch (error) {
    console.error(error);
    self.postMessage({ error: "Couldn’t analyze this photo. Check your connection and try a JPEG, PNG, or WebP file." });
  }
};
