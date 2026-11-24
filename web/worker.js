import { predict } from "./inference.js?v=3";

import { warm } from "./model-loader.js";

let latest;
self.onmessage = async ({ data }) => {
  if (data.warm) {
    warm().catch(() => {}); // A later prediction retries interrupted downloads.
    return;
  }
  const { file, id } = data;
  latest = id;
  const current = () => id === latest;
  const send = data => { if (current()) self.postMessage({ ...data, id }); };
  try {
    const result = await predict(file, status => send({ status }),
      ({ matches, seconds }) => send({ matches, seconds, stage: "quick" }), current);
    if (result) send({ matches: result.matches, seconds: result.seconds, stage: "exact" });
  } catch (error) {
    console.error(error);
    send({ error: "Couldn’t finish the analysis. Check your connection and retry with a JPG, PNG, or WebP photo." });
  }
};
