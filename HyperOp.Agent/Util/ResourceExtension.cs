using System.Reflection;

namespace HyperOp.Agent.Util
{
    public static class ResourceExtension
    {
        extension(Assembly asm)
        {
            public string ReadEmbeddedTextFile(string resourceName)
            {
                using var stream = asm.GetManifestResourceStream(resourceName);

                if (stream is null)
                {
                    throw new FileNotFoundException($"Could not find embedded resource: '{resourceName}'");
                }

                using var reader = new StreamReader(stream);

                return reader.ReadToEnd();
            }
        }

    }
}
